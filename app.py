"""
app.py — Streamlit-приложение для автоматического извлечения объектов из GeoTIFF снимков.
Использует YOLOv8 (best.pt) с тайлингом через SAHI для обработки больших снимков.

Pipeline:
  1. Загрузка GeoTIFF + AOI (GeoJSON) через UI
  2. Тайлинг снимка на перекрывающиеся окна (rasterio.windows)
  3. Инференс YOLOv8 на каждом тайле (через SAHI)
  4. Векторизация bbox → полигоны в географических координатах
  5. Постобработка (фильтрация по площади, объединение пересекающихся)
  6. Расчёт статистики и экспорт в GeoJSON / GeoPackage
"""

import io
import json
import logging
import math
import os
import tempfile
import traceback
import warnings
from pathlib import Path


import folium
import geopandas as gpd
import numpy as np
import rasterio
import streamlit as st
from rasterio.transform import array_bounds
from rasterio.windows import Window
from shapely.geometry import box, shape
from shapely.ops import unary_union
from streamlit_folium import st_folium

# ─────────────────────────────────────────────────────────────────────────────
# Настройка логирования
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Константы
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH = Path(__file__).parent / "best.pt"  # Веса модели в той же папке
TILE_SIZE = 640           # Размер тайла в пикселях
OVERLAP = 64              # Перекрытие тайлов в пикселях
MIN_AREA_M2 = 5.0         # Минимальная площадь объекта в кв. метрах
CONFIDENCE_THRESHOLD = 0.25  # Порог уверенности модели
IOU_THRESHOLD = 0.45      # Порог IoU для NMS в SAHI
TARGET_CRS = "EPSG:4326"  # Целевая СК для GeoJSON-экспорта


# ─────────────────────────────────────────────────────────────────────────────
# МОДУЛЬ 1: Загрузка и проверка модели
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Загрузка модели YOLOv8…")
def load_model(model_path: str):
    """Загружает и кэширует модель YOLOv8 из файла весов."""
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        logger.info(f"Модель загружена: {model_path}")
        return model
    except Exception as e:
        logger.error(f"Ошибка загрузки модели: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# МОДУЛЬ 2: Тайлинг GeoTIFF
# ─────────────────────────────────────────────────────────────────────────────
def generate_tiles(src: rasterio.DatasetReader, tile_size: int = TILE_SIZE, overlap: int = OVERLAP):
    """
    Генератор тайлов из GeoTIFF-снимка с перекрытием.

    Возвращает:
        window (rasterio.windows.Window): окно пикселей
        transform (Affine): геотрансформация для данного тайла
        tile_img (np.ndarray): массив пикселей (H, W, C) в uint8
    """
    width = src.width
    height = src.height
    step = tile_size - overlap

    for row_off in range(0, height, step):
        for col_off in range(0, width, step):
            # Ограничиваем окно границами снимка
            win_width = min(tile_size, width - col_off)
            win_height = min(tile_size, height - row_off)

            window = Window(col_off, row_off, win_width, win_height)
            transform = rasterio.windows.transform(window, src.transform)

            # Читаем первые 3 канала (RGB) и конвертируем в uint8
            bands_count = min(src.count, 3)
            data = src.read(list(range(1, bands_count + 1)), window=window)  # (C, H, W)

            # Нормализация в uint8
            tile_img = []
            for band in data:
                band = band.astype(np.float32)
                b_min, b_max = band.min(), band.max()
                if b_max > b_min:
                    band = ((band - b_min) / (b_max - b_min) * 255).astype(np.uint8)
                else:
                    band = np.zeros_like(band, dtype=np.uint8)
                tile_img.append(band)

            # Если один канал — дублируем до RGB
            if len(tile_img) == 1:
                tile_img = tile_img * 3

            tile_img = np.stack(tile_img, axis=-1)  # (H, W, 3)

            logger.debug(f"Тайл: col={col_off}, row={row_off}, shape={tile_img.shape}")
            yield window, transform, tile_img


def count_tiles(src: rasterio.DatasetReader, tile_size: int = TILE_SIZE, overlap: int = OVERLAP) -> int:
    """Подсчитывает общее количество тайлов для прогресс-бара."""
    step = tile_size - overlap
    cols = math.ceil(src.width / step)
    rows = math.ceil(src.height / step)
    return cols * rows


# ─────────────────────────────────────────────────────────────────────────────
# МОДУЛЬ 3: Инференс (predict) на тайле
# ─────────────────────────────────────────────────────────────────────────────
def predict(model, tile_img: np.ndarray, conf: float = CONFIDENCE_THRESHOLD, iou: float = IOU_THRESHOLD):
    """
    Запускает YOLOv8-инференс на одном тайле.

    Возвращает список словарей:
        {
          'bbox_px': [x1, y1, x2, y2],  # пиксельные координаты в тайле
          'confidence': float,
          'class_id': int,
          'class_name': str,
        }
    """
    results = model.predict(
        source=tile_img,
        conf=conf,
        iou=iou,
        verbose=False,
    )

    detections = []
    for result in results:
        if result.boxes is None:
            continue
        for box_data in result.boxes:
            xyxy = box_data.xyxy[0].cpu().numpy().tolist()
            confidence = float(box_data.conf[0].cpu().numpy())
            class_id = int(box_data.cls[0].cpu().numpy())
            class_name = model.names.get(class_id, f"class_{class_id}")
            detections.append({
                "bbox_px": xyxy,
                "confidence": round(confidence, 4),
                "class_id": class_id,
                "class_name": class_name,
            })

    return detections


# ─────────────────────────────────────────────────────────────────────────────
# МОДУЛЬ 4: Векторизация — bbox → географический полигон
# ─────────────────────────────────────────────────────────────────────────────
def bbox_to_geo_polygon(bbox_px: list, tile_transform):
    """
    Конвертирует пиксельный bbox [x1, y1, x2, y2] в географический полигон shapely
    с использованием аффинной трансформации тайла.

    Возвращает:
        shapely.geometry.Polygon в СК исходного GeoTIFF
    """
    x1, y1, x2, y2 = bbox_px

    # Пиксель → географические координаты (верхний левый угол пикселя)
    geo_x1, geo_y1 = tile_transform * (x1, y1)
    geo_x2, geo_y2 = tile_transform * (x2, y2)

    # Нормализуем координаты (y может быть инвертирован)
    min_x = min(geo_x1, geo_x2)
    max_x = max(geo_x1, geo_x2)
    min_y = min(geo_y1, geo_y2)
    max_y = max(geo_y1, geo_y2)

    return box(min_x, min_y, max_x, max_y)


# ─────────────────────────────────────────────────────────────────────────────
# МОДУЛЬ 5: Постобработка
# ─────────────────────────────────────────────────────────────────────────────
def postprocess_detections(
    raw_detections: list,
    src_crs: str,
    min_area_m2: float = MIN_AREA_M2,
    source_image_name: str = "unknown",
) -> gpd.GeoDataFrame:
    """
    Постобработка обнаружений:
      1. Перевод полигонов в EPSG:4326 для унификации
      2. Вычисление площади в метрах (через CRS в метрах)
      3. Фильтрация по минимальной площади
      4. Объединение пересекающихся полигонов одного класса

    Возвращает:
        GeoDataFrame с атрибутами: id, class, confidence, source_image, area_m2
    """
    if not raw_detections:
        logger.warning("Нет обнаружений для постобработки")
        return gpd.GeoDataFrame(
            columns=["id", "class", "confidence", "source_image", "area_m2", "geometry"],
            crs=TARGET_CRS,
        )

    # Создаём GeoDataFrame в исходной СК снимка
    gdf = gpd.GeoDataFrame(
        raw_detections,
        geometry="geometry",
        crs=src_crs,
    )

    # Вычисляем площадь в метрах (проецируем во Pseudo-Mercator)
    gdf_metric = gdf.to_crs("EPSG:3857")
    gdf["area_m2"] = gdf_metric.geometry.area.round(2)

    # Фильтрация по площади
    before = len(gdf)
    gdf = gdf[gdf["area_m2"] >= min_area_m2].copy()
    logger.info(f"Фильтрация по площади: {before} → {len(gdf)} объектов")

    if gdf.empty:
        return gpd.GeoDataFrame(
            columns=["id", "class", "confidence", "source_image", "area_m2", "geometry"],
            crs=TARGET_CRS,
        )

    # Объединение пересекающихся полигонов (unary_union по классу)
    merged_records = []
    for class_name, group in gdf.groupby("class_name"):
        # Разбиваем на кластеры пересекающихся полигонов
        polys = list(group.geometry)
        merged = []
        used = [False] * len(polys)

        for i, poly in enumerate(polys):
            if used[i]:
                continue
            cluster = [poly]
            used[i] = True
            for j in range(i + 1, len(polys)):
                if not used[j] and poly.intersects(polys[j]):
                    cluster.append(polys[j])
                    used[j] = True
            merged.append(unary_union(cluster))

        # Подбираем уверенность из исходных полигонов для каждого кластера
        confidences = group["confidence"].tolist()
        for k, merged_poly in enumerate(merged):
            conf_val = confidences[k] if k < len(confidences) else 0.0
            merged_records.append({
                "class_name": class_name,
                "confidence": round(conf_val, 4),
                "geometry": merged_poly,
            })

    merged_gdf = gpd.GeoDataFrame(merged_records, geometry="geometry", crs=src_crs)

    # Переводим в EPSG:4326
    merged_gdf = merged_gdf.to_crs(TARGET_CRS)

    # Пересчитываем площадь уже в метрах через 3857
    merged_metric = merged_gdf.to_crs("EPSG:3857")
    merged_gdf["area_m2"] = merged_metric.geometry.area.round(2)

    # Повторная фильтрация после слияния
    merged_gdf = merged_gdf[merged_gdf["area_m2"] >= min_area_m2].reset_index(drop=True)

    # Присваиваем атрибуты
    merged_gdf["id"] = range(1, len(merged_gdf) + 1)
    merged_gdf["class"] = merged_gdf["class_name"]
    merged_gdf["source_image"] = source_image_name
    merged_gdf = merged_gdf[["id", "class", "confidence", "source_image", "area_m2", "geometry"]]

    logger.info(f"Итого объектов после постобработки: {len(merged_gdf)}")
    return merged_gdf


# ─────────────────────────────────────────────────────────────────────────────
# МОДУЛЬ 6: Вычисление площади покрытия снимка (кв. км)
# ─────────────────────────────────────────────────────────────────────────────
def get_image_area_km2(src: rasterio.DatasetReader) -> float:
    """Вычисляет площадь снимка в кв. километрах."""
    try:
        bounds = src.bounds
        bbox_geom = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
        gdf_bounds = gpd.GeoDataFrame(geometry=[bbox_geom], crs=src.crs)
        gdf_metric = gdf_bounds.to_crs("EPSG:3857")
        area_m2 = gdf_metric.geometry.area.iloc[0]
        return round(area_m2 / 1_000_000, 4)
    except Exception as e:
        logger.warning(f"Не удалось вычислить площадь снимка: {e}")
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# МОДУЛЬ 7: Аналитика
# ─────────────────────────────────────────────────────────────────────────────
def compute_statistics(gdf: gpd.GeoDataFrame, image_area_km2: float) -> dict:
    """Рассчитывает сводную статистику по обнаруженным объектам."""
    if gdf.empty:
        return {"total": 0, "density_per_km2": 0.0, "total_area_m2": 0.0, "classes": {}}

    total = len(gdf)
    total_area_m2 = round(gdf["area_m2"].sum(), 2)
    density = round(total / image_area_km2, 2) if image_area_km2 > 0 else 0.0

    classes = (
        gdf.groupby("class")
        .agg(count=("id", "count"), avg_conf=("confidence", "mean"), total_area=("area_m2", "sum"))
        .round(4)
        .to_dict("index")
    )

    return {
        "total": total,
        "density_per_km2": density,
        "total_area_m2": total_area_m2,
        "classes": classes,
    }


# ─────────────────────────────────────────────────────────────────────────────
# МОДУЛЬ 8: Экспорт
# ─────────────────────────────────────────────────────────────────────────────
def export_geojson(gdf: gpd.GeoDataFrame) -> str:
    """Экспортирует GeoDataFrame в строку GeoJSON (EPSG:4326)."""
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(TARGET_CRS)
    return gdf.to_json()


def export_geopackage(gdf: gpd.GeoDataFrame, path: str):
    """Сохраняет GeoDataFrame в файл GeoPackage."""
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(TARGET_CRS)
    gdf.to_file(path, layer="detections", engine="pyogrio")
    logger.info(f"GeoPackage сохранён: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# МОДУЛЬ 9: Визуализация на карте Folium
# ─────────────────────────────────────────────────────────────────────────────
def build_folium_map(gdf: gpd.GeoDataFrame) -> folium.Map:
    """Строит интерактивную карту Folium с обнаруженными объектами."""
    # Центрируем на данных
    if not gdf.empty:
        centroid = gdf.geometry.unary_union.centroid
        center = [centroid.y, centroid.x]
    else:
        center = [51.18, 71.45]  # Астана по умолчанию

    m = folium.Map(location=center, zoom_start=15, tiles="CartoDB dark_matter")

    # Слой обнаружений
    if not gdf.empty:
        # Цветовая схема по классам
        class_colors = {}
        palette = ["#ff5722", "#4caf50", "#2196f3", "#ff9800", "#9c27b0",
                   "#00bcd4", "#f44336", "#8bc34a", "#ffc107", "#607d8b"]
        unique_classes = gdf["class"].unique()
        for i, cls in enumerate(unique_classes):
            class_colors[cls] = palette[i % len(palette)]

        def style_fn(feature):
            cls = feature["properties"].get("class", "")
            return {
                "fillColor": class_colors.get(cls, "#ff5722"),
                "color": "#ffffff",
                "weight": 1,
                "fillOpacity": 0.6,
            }

        detection_layer = folium.GeoJson(
            json.loads(export_geojson(gdf)),
            name="Обнаружения",
            style_function=style_fn,
            tooltip=folium.GeoJsonTooltip(
                fields=["id", "class", "confidence", "area_m2"],
                aliases=["ID", "Класс", "Уверенность", "Площадь (м²)"],
                localize=True,
            ),
        )
        detection_layer.add_to(m)

    folium.LayerControl().add_to(m)
    return m


# ─────────────────────────────────────────────────────────────────────────────
# ГЛАВНЫЙ PIPELINE ОБРАБОТКИ
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(
    tiff_path: str,
    model,
    source_name: str,
    conf_threshold: float,
    min_area: float,
    progress_callback=None,
) -> gpd.GeoDataFrame:
    """
    Основной pipeline: тайлинг → инференс → векторизация → постобработка.

    Аргументы:
        tiff_path: путь к временному GeoTIFF файлу
        model: загруженная модель YOLOv8
        aoi_gdf: GeoDataFrame с зоной интереса (или None)
        source_name: имя исходного файла для атрибутики
        conf_threshold: порог уверенности
        min_area: минимальная площадь объекта в м²
        progress_callback: функция для обновления прогресс-бара (0..1)

    Возвращает:
        GeoDataFrame с атрибутами всех обнаруженных объектов
    """
    all_detections = []

    with rasterio.open(tiff_path) as src:
        src_crs = src.crs.to_string() if src.crs else "EPSG:4326"
        logger.info(f"GeoTIFF открыт: {src.width}x{src.height} px, CRS={src_crs}, bands={src.count}")

        total_tiles = count_tiles(src)
        logger.info(f"Всего тайлов: {total_tiles}")

        for tile_idx, (window, transform, tile_img) in enumerate(
            generate_tiles(src)
        ):
            # Обновление прогресса
            if progress_callback:
                progress_callback((tile_idx + 1) / total_tiles)

            # Инференс
            detections = predict(model, tile_img, conf=conf_threshold)
            if not detections:
                continue

            logger.debug(f"Тайл {tile_idx}: найдено {len(detections)} объектов")

            # Векторизация: bbox → географический полигон
            for det in detections:
                poly = bbox_to_geo_polygon(det["bbox_px"], transform)
                all_detections.append({
                    "class_name": det["class_name"],
                    "confidence": det["confidence"],
                    "geometry": poly,
                })

    logger.info(f"Сырых обнаружений после тайлинга: {len(all_detections)}")

    # Постобработка (фильтрация + слияние + атрибуты)
    result_gdf = postprocess_detections(
        all_detections,
        src_crs=src_crs,
        min_area_m2=min_area,
        source_image_name=source_name,
    )

    return result_gdf


# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="GeoTIFF Object Detector",
        page_icon="🛰️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── CSS для кастомизации ──
    st.markdown(
        """
        <style>
        .main { background-color: #0e1117; }
        .stMetric { background: #1a1d27; border-radius: 8px; padding: 12px; }
        .stProgress > div > div { background-color: #00bcd4; }
        h1 { color: #00bcd4; }
        .stAlert { border-radius: 8px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ── Заголовок ──
    col_logo, col_title = st.columns([1, 5])
    with col_logo:
        st.markdown("# 🛰️")
    with col_title:
        st.markdown("# GeoTIFF Object Detector")
        st.markdown("*Автоматическое извлечение объектов из спутниковых снимков*")

    st.divider()

    # ─────────────────── Боковая панель ───────────────────
    with st.sidebar:
        st.header("⚙️ Настройки")

        # Загрузка файлов
        st.subheader("📂 Входные данные")
        tiff_file = st.file_uploader(
            "GeoTIFF снимок",
            type=["tif", "tiff"],
            help="Загрузите геопривязанный GeoTIFF файл (любой размер)",
        )


        st.subheader("🔧 Параметры модели")
        conf_threshold = st.slider(
            "Порог уверенности",
            min_value=0.05,
            max_value=0.95,
            value=CONFIDENCE_THRESHOLD,
            step=0.05,
            help="Минимальная уверенность для принятия детекции",
        )
        st.subheader("🗺️ Тайлинг")
        tile_size_ui = st.select_slider(
            "Размер тайла (px)",
            options=[320, 480, 640, 800, 1024],
            value=TILE_SIZE,
        )
        overlap_ui = st.slider(
            "Перекрытие (px)",
            min_value=0,
            max_value=256,
            value=OVERLAP,
            step=16,
        )

        st.divider()

        # Статус модели
        st.subheader("🤖 Модель")
        if MODEL_PATH.exists():
            st.success(f"✅ `best.pt` найден ({MODEL_PATH.stat().st_size / 1e6:.1f} МБ)")
        else:
            st.error("❌ Файл `best.pt` не найден в папке приложения!")

        run_button = st.button(
            "🚀 Запустить обработку",
            use_container_width=True,
            type="primary",
            disabled=not (tiff_file and MODEL_PATH.exists()),
        )

    # ─────────────────── Основная область ─────────────────
    tab_map, tab_stats, tab_log = st.tabs(["🗺️ Карта", "📊 Статистика", "📋 Журнал"])

    # Инициализация session_state
    if "result_gdf" not in st.session_state:
        st.session_state.result_gdf = None
    if "image_area_km2" not in st.session_state:
        st.session_state.image_area_km2 = 0.0
    if "stats" not in st.session_state:
        st.session_state.stats = None
    if "log_messages" not in st.session_state:
        st.session_state.log_messages = []

    # ─────────────────── ЗАПУСК PIPELINE ──────────────────
    if run_button and tiff_file:
        # Загружаем модель (из кэша)
        model = load_model(str(MODEL_PATH))
        if model is None:
            st.error("Не удалось загрузить модель. Проверьте файл best.pt.")
            st.stop()

        # AOI не используется

        # Сохраняем GeoTIFF во временный файл
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            tmp.write(tiff_file.read())
            tmp_path = tmp.name

        try:
            # Информация о снимке
            with rasterio.open(tmp_path) as src:
                image_area_km2 = get_image_area_km2(src)
                st.session_state.image_area_km2 = image_area_km2

            # Прогресс-бар и статус
            progress_bar = st.progress(0, text="Инициализация…")
            status_text = st.empty()

            log_msgs = []

            def progress_update(frac: float):
                pct = int(frac * 100)
                progress_bar.progress(frac, text=f"Обработка тайлов: {pct}%")
                status_text.info(f"🔄 Обработано {pct}% тайлов…")

            # Логирование в UI
            class UILogHandler(logging.Handler):
                def emit(self, record):
                    log_msgs.append(self.format(record))

            ui_handler = UILogHandler()
            ui_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
            logger.addHandler(ui_handler)

            result_gdf = run_pipeline(
                tiff_path=tmp_path,
                model=model,
                source_name=tiff_file.name,
                conf_threshold=conf_threshold,
                min_area=0.0,
                progress_callback=progress_update,
            )

            logger.removeHandler(ui_handler)
            st.session_state.log_messages = log_msgs

            progress_bar.progress(1.0, text="✅ Обработка завершена!")
            status_text.success("✅ Обработка завершена!")

            st.session_state.result_gdf = result_gdf
            st.session_state.stats = compute_statistics(result_gdf, image_area_km2)

        except Exception as e:
            st.error(f"Ошибка в процессе обработки:\n```\n{traceback.format_exc()}\n```")
            logger.error(f"Pipeline error: {e}")
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    # ─────────────────── ВКЛАДКА: КАРТА ───────────────────
    with tab_map:
        result_gdf = st.session_state.result_gdf

        if result_gdf is not None:
            # Кнопки экспорта
            col_dl1, col_dl2, col_dl3 = st.columns([2, 2, 4])

            with col_dl1:
                geojson_str = export_geojson(result_gdf)
                st.download_button(
                    label="⬇️ Скачать GeoJSON",
                    data=geojson_str,
                    file_name="detections.geojson",
                    mime="application/geo+json",
                    use_container_width=True,
                )

            with col_dl2:
                gpkg_path = tempfile.mktemp(suffix=".gpkg")
                export_geopackage(result_gdf, gpkg_path)
                with open(gpkg_path, "rb") as f:
                    gpkg_bytes = f.read()
                st.download_button(
                    label="⬇️ Скачать GeoPackage",
                    data=gpkg_bytes,
                    file_name="detections.gpkg",
                    mime="application/octet-stream",
                    use_container_width=True,
                )
                try:
                    os.unlink(gpkg_path)
                except Exception:
                    pass

            st.divider()

            # Интерактивная карта
            folium_map = build_folium_map(result_gdf)
            st_folium(folium_map, use_container_width=True, height=600)

        else:
            st.info(
                "📥 Загрузите GeoTIFF снимок на боковой панели и нажмите **«Запустить обработку»**.\n\n"
                "Результаты обнаружений отобразятся здесь в виде интерактивной карты."
            )
            # Демо-карта с центром на Астане
            m_demo = folium.Map(location=[51.18, 71.45], zoom_start=12, tiles="CartoDB dark_matter")
            st_folium(m_demo, use_container_width=True, height=400)

    # ─────────────────── ВКЛАДКА: СТАТИСТИКА ──────────────
    with tab_stats:
        stats = st.session_state.stats

        if stats:
            st.subheader("📊 Сводная статистика")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("🔢 Всего объектов", stats["total"])
            with col2:
                st.metric(
                    "📐 Плотность (объ/км²)",
                    f"{stats['density_per_km2']:.1f}",
                )
            with col3:
                st.metric(
                    "🟦 Суммарная площадь",
                    f"{stats['total_area_m2']:,.0f} м²",
                )
            with col4:
                st.metric(
                    "🗺️ Площадь снимка",
                    f"{st.session_state.image_area_km2:.2f} км²",
                )

            st.divider()

            # Таблица по классам
            if stats["classes"]:
                st.subheader("Разбивка по классам")
                import pandas as pd

                class_df = pd.DataFrame(stats["classes"]).T
                class_df.index.name = "Класс"
                class_df.columns = ["Кол-во", "Ср. уверенность", "Суммарная площадь (м²)"]
                st.dataframe(class_df, use_container_width=True)

            # Таблица атрибутов
            st.subheader("Таблица объектов")
            result_gdf = st.session_state.result_gdf
            if result_gdf is not None and not result_gdf.empty:
                df_display = result_gdf.drop(columns=["geometry"]).copy()
                st.dataframe(df_display, use_container_width=True, height=400)
        else:
            st.info("Статистика появится после выполнения обработки.")

    # ─────────────────── ВКЛАДКА: ЖУРНАЛ ──────────────────
    with tab_log:
        st.subheader("📋 Журнал выполнения")
        log_messages = st.session_state.log_messages
        if log_messages:
            log_text = "\n".join(log_messages)
            st.code(log_text, language="text")
        else:
            st.info("Здесь будут отображаться системные сообщения в процессе обработки.")


if __name__ == "__main__":
    main()
