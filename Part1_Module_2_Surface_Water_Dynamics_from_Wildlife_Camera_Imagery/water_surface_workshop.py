from datetime import datetime
from pathlib import Path
import re

import numpy as np
import pandas as pd
from PIL import Image, ImageOps


EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}


def image_paths(folder):
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f'Image folder not found: {folder}')
    paths = sorted(path for path in folder.iterdir()
                   if path.is_file() and path.suffix.lower() in EXTENSIONS)
    if not paths:
        raise ValueError(f'No images found in {folder}')
    return paths


def load_rgb(path):
    with Image.open(path) as source:
        return ImageOps.exif_transpose(source).convert('RGB')


def grayscale_fraction(image, channel_tolerance=10):
    if not 0 <= channel_tolerance <= 255:
        raise ValueError('Channel tolerance must be between 0 and 255.')
    pixels = np.asarray(image.convert('RGB'), dtype=np.int16)
    spread = pixels.max(axis=2) - pixels.min(axis=2)
    return float(np.mean(spread <= channel_tolerance))


def preprocess_case1(folder, width=640, channel_tolerance=10,
                     grayscale_cutoff=0.98, output_dir=None):
    if width <= 0 or not 0 <= grayscale_cutoff <= 1:
        raise ValueError('Width must be positive; cutoff must be in [0, 1].')
    folder = Path(folder)
    output = Path(output_dir) if output_dir is not None else folder / 'preprocess'
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for path in image_paths(folder):
        image = load_rgb(path)
        fraction = grayscale_fraction(image, channel_tolerance)
        rejected = fraction >= grayscale_cutoff
        destination = output / f'{path.name}.png'
        height = max(1, round(image.height * width / image.width))
        if rejected:
            destination.unlink(missing_ok=True)
        else:
            image.resize((width, height), Image.Resampling.LANCZOS).save(destination)
        records.append(dict(filename=path.name, grayscale_fraction=fraction,
                            status='night_vision' if rejected else 'kept',
                            original_width=image.width, original_height=image.height,
                            output_width=None if rejected else width,
                            output_height=None if rejected else height,
                            output_path='' if rejected else str(destination)))
    report = pd.DataFrame(records)
    report.to_csv(output / 'preprocess_report.csv', index=False)
    return report


def load_sam3(model_id='facebook/sam3'):
    import torch
    from transformers import Sam3Model, Sam3Processor

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Loading {model_id} on {device}')
    model = Sam3Model.from_pretrained(model_id).to(device).eval()
    processor = Sam3Processor.from_pretrained(model_id)
    return model, processor, device


def segment_water(image, model, processor, device, text='water',
                  detection_threshold=0.5, mask_threshold=0.5):
    import torch

    inputs = processor(images=image, text=text, return_tensors='pt')
    inputs = {key: value.to(device) if isinstance(value, torch.Tensor) else value
              for key, value in inputs.items()}
    with torch.inference_mode():
        outputs = model(**inputs)
    result = processor.post_process_instance_segmentation(
        outputs, threshold=detection_threshold, mask_threshold=mask_threshold,
        target_sizes=inputs['original_sizes'].tolist(),
    )[0]
    combined = np.zeros((image.height, image.width), dtype=bool)
    for mask in result['masks']:
        values = mask.detach().cpu().numpy() if isinstance(mask, torch.Tensor) else np.asarray(mask)
        values = np.squeeze(values)
        if values.shape != combined.shape:
            raise ValueError(f'Unexpected mask shape: {values.shape}')
        combined |= values > 0
    return combined


def overlay_mask(image, mask, color=(255, 40, 40), alpha=0.45):
    pixels = np.asarray(image).copy()
    if pixels.shape[:2] != mask.shape:
        raise ValueError('Image and mask dimensions differ.')
    pixels[mask] = ((1 - alpha) * pixels[mask] + alpha * np.asarray(color)).astype(np.uint8)
    return Image.fromarray(pixels)


def timestamp_from_name(filename):
    match = re.search(r'_(\d{8})_(\d{6})PST(?:-|\.)', filename)
    if match is None:
        raise ValueError(f'No YYYYMMDD_HHMMSS PST timestamp in {filename}')
    return datetime.strptime(''.join(match.groups()), '%Y%m%d%H%M%S')


def segment_folder(folder, output, model, processor, device, text='water',
                   detection_threshold=0.5, mask_threshold=0.5):
    paths = image_paths(folder)
    output = Path(output)
    for name in ('masks', 'overlays'):
        (output / name).mkdir(parents=True, exist_ok=True)
    records = []
    for number, path in enumerate(paths, start=1):
        image = load_rgb(path)
        if image.width != 640:
            raise ValueError(f'{path.name}: expected width 640, got {image.width}')
        mask = segment_water(image, model, processor, device, text,
                             detection_threshold, mask_threshold)
        mask_path = output / 'masks' / f'{path.name}.npy'
        np.save(mask_path, mask)
        overlay_mask(image, mask).save(output / 'overlays' / f'{path.name}.png')
        records.append(dict(filename=path.name, image_path=str(path),
                            timestamp=timestamp_from_name(path.name),
                            mask_path=str(mask_path), water_pixels=int(mask.sum()),
                            image_pixels=mask.size, water_fraction=float(mask.mean()),
                            detection_status='water' if mask.any() else 'no_detection'))
        print(f'[{number}/{len(paths)}] {path.name}: {mask.mean():.4f}', flush=True)
    table = pd.DataFrame(records)
    table.to_csv(output / 'segmentation_summary.csv', index=False)
    return table


def largest_water_mask(table):
    if table.empty:
        raise ValueError('No segmentation results.')
    record = table.loc[table['water_fraction'].idxmax()]
    mask = np.load(record['mask_path'], allow_pickle=False).astype(bool)
    if not mask.any():
        raise ValueError('All masks are empty. Inspect the prompt, thresholds, and overlays.')
    return record, mask


def expand_horizontal(mask, pixels=0):
    if not isinstance(pixels, (int, np.integer)) or pixels < 0:
        raise ValueError('Expansion must be a nonnegative integer.')
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError('Expected a two-dimensional mask.')
    width = mask.shape[1]
    padding = min(int(pixels), width)
    padded = np.pad(mask.astype(np.int32), ((0, 0), (padding, padding)))
    cumulative = np.pad(np.cumsum(padded, axis=1), ((0, 0), (1, 0)))
    window = 2 * padding + 1
    return (cumulative[:, window:] - cumulative[:, :-window]) > 0


def row_fractions(mask, reference):
    mask = np.asarray(mask, dtype=bool)
    reference = np.asarray(reference, dtype=bool)
    if mask.ndim != 2 or mask.shape != reference.shape:
        raise ValueError('Masks must be two-dimensional and have the same shape.')
    denominator = reference.sum(axis=1)
    valid = denominator > 0
    if not valid.any():
        raise ValueError('Reference mask is empty.')
    fractions = np.full(mask.shape[0], np.nan)
    fractions[valid] = (mask & reference).sum(axis=1)[valid] / denominator[valid]
    return fractions


def calculate_time_series(table, reference, output, expansion_pixels=0):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    table = table.copy()
    row_values = []
    for path in table['mask_path']:
        row_values.append(row_fractions(np.load(path, allow_pickle=False), reference))
    values = np.asarray(row_values)
    row_table = pd.DataFrame(values, columns=[f'row_{row}' for row in range(reference.shape[0])])
    row_table.insert(0, 'filename', table['filename'].to_numpy())
    row_table.sort_values('filename').to_csv(output / 'row_fractions.csv', index=False)
    np.save(output / 'reference_mask.npy', reference)
    table['mean_row_fraction'] = np.nanmean(values, axis=1)
    table['reference_rows'] = int(np.count_nonzero(reference.any(axis=1)))
    table['expansion_pixels'] = expansion_pixels
    table = table.sort_values(['timestamp', 'filename'])
    grouped = table.groupby('timestamp', as_index=False).agg(
        mean_row_fraction=('mean_row_fraction', 'mean'), photo_count=('filename', 'size'))
    table.to_csv(output / 'photo_fractions.csv', index=False)
    grouped.to_csv(output / 'timestamp_fractions.csv', index=False)
    return table, grouped, values


def plot_time_series(table, grouped, output):
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    figure, axis = plt.subplots(figsize=(12, 4))
    axis.scatter(table['timestamp'], table['mean_row_fraction'], s=12, alpha=0.3,
                 color='gray', label='Individual photos')
    axis.plot(grouped['timestamp'], grouped['mean_row_fraction'], '.-',
              color='teal', linewidth=1, label='Mean per timestamp')
    locator = mdates.AutoDateLocator()
    axis.xaxis.set_major_locator(locator)
    axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    axis.set(xlabel='Camera time (PST label in filename)',
             ylabel='Mean row water fraction', ylim=(-0.02, 1.02))
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(Path(output) / 'water_fraction_vs_time.png', dpi=180)
    return figure