//! Shared tqdm-style presentation for terminals, IDEs and redirected stderr.
use std::collections::BTreeMap;
use std::io::{self, IsTerminal, Write};
use std::time::{Duration, Instant};
pub fn stage_name(stage: &str) -> &str {
    match stage {
        "输入分类" => "classify",
        "原生全文分析" => "native_analysis",
        "原生分析失败，回退版面模型" => "layout_fallback",
        "读取文字与渲染" => "render",
        "版面模型识别" => "layout",
        "公式识别" => "formula",
        "正文 OCR 检测与原生文字规划" => "ocr_detection",
        "正文 OCR 批量识别" => "ocr_recognition",
        "表格识别" => "table",
        "印章识别" => "seal",
        "页面组装与图片资产" => "assemble",
        "补识别、段落和跨页表格" => "finalize",
        "写出结果" => "write",
        _ => stage,
    }
}
fn label(stage: &str) -> &str {
    match stage {
        "layout" => "Layout Predict",
        "formula" => "Formula Predict",
        "ocr_detection" => "PDF/OCR-det Predict",
        "ocr_recognition" => "OCR-rec Predict",
        "table" => "Table Predict",
        "assemble" => "Processing pages",
        "orientation" => "Orientation Predict",
        "seal" => "Seal Predict",
        "finalize" => "Post-processing",
        _ => stage,
    }
}
fn clock(seconds: f64) -> String {
    let s = seconds.max(0.) as u64;
    if s >= 3600 {
        format!("{}:{:02}:{:02}", s / 3600, (s / 60) % 60, s % 60)
    } else {
        format!("{:02}:{:02}", s / 60, s % 60)
    }
}
pub fn format_stage(stage: &str, done: usize, total: usize, elapsed: Duration) -> String {
    let done = done.min(total);
    let fill = if total == 0 {
        20
    } else {
        done.saturating_mul(20) / total
    };
    let percent = if total == 0 {
        100
    } else {
        done.saturating_mul(100) / total
    };
    let seconds = elapsed.as_secs_f64();
    let (eta, rate) = if done > 0 && seconds > 0. {
        let rate = done as f64 / seconds;
        (
            clock((total - done) as f64 / rate),
            if rate >= 1. {
                format!("{rate:.2}it/s")
            } else {
                format!("{:.2}s/it", 1. / rate)
            },
        )
    } else if total == 0 {
        ("00:00".into(), "0.00it/s".into())
    } else {
        ("?".into(), "?it/s".into())
    };
    format!(
        "{}: {:3}%|{}{}| {}/{} [{}<{}, {}]",
        label(stage),
        percent,
        "█".repeat(fill),
        " ".repeat(20 - fill),
        done,
        total,
        clock(seconds),
        eta,
        rate
    )
}
struct Stage {
    measured: Option<Duration>,
    done: usize,
    total: usize,
    started: Instant,
    finished: bool,
}
pub struct TerminalProgress {
    enabled: bool,
    dynamic: bool,
    stages: BTreeMap<String, Stage>,
    last_draw: Option<Instant>,
    width: usize,
    active: Option<String>,
}
impl TerminalProgress {
    pub fn new(_name: String, _total: usize) -> Self {
        let mode = std::env::var("RAPIDDOC_PROGRESS")
            .unwrap_or_default()
            .to_ascii_lowercase();
        let dynamic = match mode.as_str() {
            "always" | "1" => true,
            "plain" => false,
            _ => io::stderr().is_terminal() || std::env::var_os("PYCHARM_HOSTED").is_some(),
        };
        Self {
            enabled: !matches!(mode.as_str(), "off" | "0"),
            dynamic,
            stages: BTreeMap::new(),
            last_draw: None,
            width: 0,
            active: None,
        }
    }
    pub fn update(&mut self, stage: &str, page: Option<usize>) {
        self.update_detail(stage, page, None)
    }
    pub fn update_detail(&mut self, stage: &str, _page: Option<usize>, _detail: Option<&str>) {
        if matches!(stage_name(stage), "finalize" | "write") {
            self.clear();
            if self.enabled {
                let _ = writeln!(io::stderr().lock(), "{}...", label(stage));
            }
        }
    }
    pub fn reset(&mut self) {
        self.clear();
        self.stages.clear();
        self.active = None;
    }
    pub fn advance(&mut self, stage: &str, done: usize, total: usize) {
        self.advance_with_elapsed(stage, done, total, None)
    }
    /// Production events carry measured stage work; gaps between pages are not work.
    pub fn advance_with_elapsed(
        &mut self,
        stage: &str,
        done: usize,
        total: usize,
        elapsed: Option<Duration>,
    ) {
        if !self.enabled || total == 0 {
            return;
        }
        let stage = stage_name(stage);
        let fresh = !self.stages.contains_key(stage);
        let state = self.stages.entry(stage.into()).or_insert_with(|| Stage {
            measured: None,
            done: 0,
            total,
            started: Instant::now(),
            finished: false,
        });
        if state.finished {
            return;
        }
        state.done = state.done.max(done.min(total));
        state.total = total;
        if let Some(elapsed) = elapsed {
            state.measured = Some(state.measured.unwrap_or_default().max(elapsed));
        }
        let finished = state.done == total;
        let due = self.last_draw.is_none_or(|t| {
            t.elapsed()
                >= if self.dynamic {
                    Duration::from_millis(100)
                } else {
                    Duration::from_secs(5)
                }
        });
        let switched = self.active.as_deref() != Some(stage);
        if !finished && !fresh && !due && !(self.dynamic && switched) {
            return;
        }
        let text = format_stage(
            stage,
            state.done,
            total,
            state.measured.unwrap_or_else(|| state.started.elapsed()),
        );
        state.finished = finished;
        let mut stderr = io::stderr().lock();
        if self.dynamic {
            let _ = write!(
                stderr,
                "\r{}{}",
                text,
                " ".repeat(self.width.saturating_sub(text.chars().count()))
            );
            if finished {
                let _ = writeln!(stderr);
                self.width = 0;
            } else {
                self.width = text.chars().count();
            }
        } else {
            let _ = writeln!(stderr, "{text}");
        }
        let _ = stderr.flush();
        self.last_draw = Some(Instant::now());
        self.active = Some(stage.into());
    }
    pub fn clear(&mut self) {
        if self.dynamic && self.width > 0 {
            let mut stderr = io::stderr().lock();
            let _ = write!(stderr, "\r{}\r", " ".repeat(self.width));
            let _ = stderr.flush();
            self.width = 0;
        }
        self.last_draw = None;
    }
}
impl Drop for TerminalProgress {
    fn drop(&mut self) {
        self.clear();
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stage_counts_eta_rate_and_zero_work() {
        let line = format_stage("formula", 2, 5, Duration::from_secs(4));
        assert!(line.starts_with("Formula Predict:  40%|"));
        assert!(line.ends_with("2/5 [00:04<00:06, 2.00s/it]"));
        assert!(format_stage("layout", 14, 14, Duration::from_secs(7))
            .ends_with("14/14 [00:07<00:00, 2.00it/s]"));
        assert!(format_stage("table", 0, 0, Duration::ZERO).contains("100%"));
        assert!(format_stage("formula", 0, 5, Duration::ZERO).contains("00:00<?, ?it/s"));
    }
}
