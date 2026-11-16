use rapiddoc_core::pillow_resize::Rgb;
use regex::Regex;
use std::{collections::HashMap, sync::LazyLock};

pub(super) fn repetitive(latex: &str) -> bool {
    static TOKEN: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"\\[A-Za-z]+|[A-Za-z0-9]+").unwrap());
    let mut counts = HashMap::new();
    let mut total = 0;
    for token in TOKEN.find_iter(latex) {
        *counts.entry(token.as_str()).or_insert(0usize) += 1;
        total += 1;
    }
    total >= 128 && counts.values().any(|count| count * 3 >= total)
}

fn neighbors(index: usize, width: usize, height: usize) -> impl Iterator<Item = usize> {
    let (x, y) = (index % width, index / width);
    (y.saturating_sub(1)..=(y + 1).min(height - 1)).flat_map(move |row| {
        (x.saturating_sub(1)..=(x + 1).min(width - 1)).map(move |col| row * width + col)
    })
}

pub(super) fn clean(image: &Rgb) -> Option<Rgb> {
    let (w, h) = (image.w, image.h);
    if w.min(h) < 32 {
        return None;
    }
    // 与 Python 的 OpenCV RGB 灰度转换及 8 连通域一致。
    let gray: Vec<_> = image
        .data
        .chunks_exact(3)
        .map(|p| {
            ((9798 * p[0] as u32 + 19235 * p[1] as u32 + 3735 * p[2] as u32 + 16384) >> 15) as u8
        })
        .collect();
    let mut unvisited: Vec<_> = gray.iter().map(|&value| value < 135).collect();
    let mut foreground = vec![false; gray.len()];
    let mut component = Vec::new();
    let mut speckles = 0;
    for start in 0..gray.len() {
        if !unvisited[start] {
            continue;
        }
        component.clear();
        component.push(start);
        unvisited[start] = false;
        let mut cursor = 0;
        while cursor < component.len() {
            for neighbor in neighbors(component[cursor], w, h) {
                if unvisited[neighbor] {
                    unvisited[neighbor] = false;
                    component.push(neighbor);
                }
            }
            cursor += 1;
        }
        if component.len() < 5 {
            speckles += 1;
        } else {
            // 膨胀一像素以保留深色笔画的浅色边缘。
            for &pixel in &component {
                for neighbor in neighbors(pixel, w, h) {
                    foreground[neighbor] = true;
                }
            }
        }
    }
    if speckles < 16 || !foreground.iter().any(|&pixel| pixel) {
        return None;
    }

    // 九个区域都必须有足量散点，避免把局部公式符号当成背景。
    let bounds = |n: usize, i: usize| n / 3 * i + i.min(n % 3);
    for row in 0..3 {
        for col in 0..3 {
            let (x0, x1) = (bounds(w, col), bounds(w, col + 1));
            let (y0, y1) = (bounds(h, row), bounds(h, row + 1));
            let count = (y0..y1)
                .flat_map(|y| (x0..x1).map(move |x| y * w + x))
                .filter(|&i| gray[i] < 220 && !foreground[i])
                .count();
            if count * 40 < (x1 - x0) * (y1 - y0) {
                return None;
            }
        }
    }
    let mut data = image.data.clone();
    for (pixel, preserve) in data.chunks_exact_mut(3).zip(foreground) {
        if !preserve {
            pixel.fill(255);
        }
    }
    Some(Rgb { data, w, h })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn removes_distributed_speckles_and_keeps_dark_strokes() {
        let mut image = Rgb {
            data: vec![255; 90 * 60 * 3],
            w: 90,
            h: 60,
        };
        let stroke = (25 * image.w + 42) * 3;
        image.data[stroke..stroke + 60].fill(0);
        assert!(clean(&image).is_none());
        for y in (0..image.h).step_by(4) {
            for x in (0..image.w).step_by(4) {
                let i = (y * image.w + x) * 3;
                image.data[i..i + 3].fill(100);
            }
        }
        let cleaned = clean(&image).unwrap();
        assert_eq!(&cleaned.data[..3], &[255; 3]);
        assert_eq!(&cleaned.data[stroke..stroke + 60], &[0; 60]);
        assert!(repetitive(&r"\mathrm{d}(".repeat(80)));
        assert!(!repetitive(r"y=\frac{1}{2}"));
    }
}
