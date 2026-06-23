//! Odds JSON loading for exacta and trifecta odds

use chrono::{DateTime, FixedOffset, TimeZone};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Exacta odds JSON structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExactaOddsFile {
    pub date: u32,
    pub stadium_code: u8,
    pub race_no: u8,
    pub scraped_at: String,
    pub exacta: HashMap<String, f64>,
}

/// Trifecta odds JSON structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrifectaOddsFile {
    pub date: u32,
    pub stadium_code: u8,
    pub race_no: u8,
    pub scraped_at: String,
    pub trifecta: HashMap<String, f64>,
}

/// Load exacta odds from JSON file
///
/// Returns a HashMap with (first, second) -> odds mapping
pub fn load_exacta_odds<P: AsRef<Path>>(
    odds_dir: P,
    date: u32,
    stadium_code: u8,
    race_no: u8,
) -> Option<HashMap<(u8, u8), f64>> {
    let filename = format!("{}_{:02}_{:02}.json", date, stadium_code, race_no);
    let path = odds_dir.as_ref().join(&filename);

    let content = fs::read_to_string(&path).ok()?;
    let odds_file: ExactaOddsFile = serde_json::from_str(&content).ok()?;

    let mut result = HashMap::new();
    for (key, odds) in odds_file.exacta {
        if let Some((first, second)) = parse_exacta_key(&key) {
            result.insert((first, second), odds);
        }
    }

    Some(result)
}

/// Load trifecta odds from JSON file
///
/// Returns a HashMap with (first, second, third) -> odds mapping
pub fn load_trifecta_odds<P: AsRef<Path>>(
    odds_dir: P,
    date: u32,
    stadium_code: u8,
    race_no: u8,
) -> Option<HashMap<(u8, u8, u8), f64>> {
    let filename = format!("{}_{:02}_{:02}_3t.json", date, stadium_code, race_no);
    let path = odds_dir.as_ref().join(&filename);

    let content = fs::read_to_string(&path).ok()?;
    let odds_file: TrifectaOddsFile = serde_json::from_str(&content).ok()?;

    let mut result = HashMap::new();
    for (key, odds) in odds_file.trifecta {
        if let Some((first, second, third)) = parse_trifecta_key(&key) {
            result.insert((first, second, third), odds);
        }
    }

    Some(result)
}

/// Outcome of selecting a pre-deadline odds snapshot for one race.
///
/// `K` is `(u8, u8)` for exacta or `(u8, u8, u8)` for trifecta. The variants are
/// fail-closed: a race yields usable odds only via [`PreDeadlineOdds::Selected`];
/// the other two variants both mean "do not bet real odds on this race".
#[derive(Debug, Clone, PartialEq)]
pub enum PreDeadlineOdds<K: Eq + std::hash::Hash> {
    /// The latest snapshot captured strictly before the race's deadline.
    Selected {
        odds: HashMap<K, f64>,
        /// RFC3339 capture time of the chosen snapshot.
        scraped_at: String,
        /// The race deadline (`HH:MM`, JST) the capture was checked against.
        deadline: String,
        /// Whether the deadline was scraped (`true`) or approximated (`false`).
        deadline_exact: bool,
    },
    /// Snapshot files exist for the race but none are verifiably pre-deadline
    /// (all captured at/after the deadline, or with an unparseable/absent
    /// deadline or timestamp). Fail-closed: treated as unusable, not bet.
    OnlyPostClose,
    /// No snapshot files exist for the race.
    NoSnapshots,
}

/// A snapshot file on disk. A local mirror of `scraper::OddsSnapshot` so the
/// always-compiled data layer never depends on the optional `scraper` feature.
/// Unused fields (stadium_code, race_no, bet_type) are intentionally omitted —
/// serde ignores the extra JSON keys.
#[derive(Debug, Clone, Deserialize)]
struct SnapshotFile {
    date: u32,
    scraped_at: String,
    deadline: Option<String>,
    #[serde(default)]
    deadline_exact: bool,
    odds: HashMap<String, f64>,
}

/// Internal result of scanning the snapshot directory for one race.
enum SnapshotSelection {
    Selected(SnapshotFile),
    OnlyPostClose,
    NoSnapshots,
}

/// Parse an RFC3339 timestamp into an absolute instant. Naive timestamps without
/// an offset (e.g. legacy `2025-12-30T20:24:02.567335`) fail here and are
/// excluded — they cannot prove they predate a deadline (fail-closed).
fn parse_scraped_at(s: &str) -> Option<DateTime<FixedOffset>> {
    DateTime::parse_from_rfc3339(s).ok()
}

/// Build the deadline instant from a race's `date` (YYYYMMDD) and `HH:MM`,
/// interpreted as JST (+09:00). Boatrace times are always JST and have no DST,
/// so a fixed +9h offset is exact — no `chrono-tz` dependency is needed.
fn deadline_instant(date: u32, deadline: &str) -> Option<DateTime<FixedOffset>> {
    let year = i32::try_from(date / 10000).ok()?;
    let month = date / 100 % 100;
    let day = date % 100;
    let (hh, mm) = deadline.split_once(':')?;
    let hour: u32 = hh.parse().ok()?;
    let minute: u32 = mm.parse().ok()?;
    let jst = FixedOffset::east_opt(9 * 3600)?;
    jst.with_ymd_and_hms(year, month, day, hour, minute, 0)
        .single()
}

/// Capture instant of a snapshot iff it is verifiably pre-deadline: it has a
/// parseable deadline and a parseable timestamp strictly before that deadline.
/// `None` (fail-closed) when the deadline or timestamp is missing/unparseable or
/// the capture is at/after the deadline.
fn predeadline_capture_instant(snap: &SnapshotFile) -> Option<DateTime<FixedOffset>> {
    let deadline_at = deadline_instant(snap.date, snap.deadline.as_deref()?)?;
    let captured_at = parse_scraped_at(&snap.scraped_at)?;
    (captured_at < deadline_at).then_some(captured_at)
}

/// Pick the latest pre-deadline snapshot from one race's candidates.
/// Empty input -> `NoSnapshots`; non-empty but none verifiably pre-deadline ->
/// `OnlyPostClose` (fail-closed). The latest survivor is chosen by parsed instant,
/// not filename order, which is unreliable across mixed RFC3339 offsets.
fn choose_predeadline(snaps: &[SnapshotFile]) -> SnapshotSelection {
    if snaps.is_empty() {
        return SnapshotSelection::NoSnapshots;
    }
    let best = snaps
        .iter()
        .filter_map(|s| predeadline_capture_instant(s).map(|at| (at, s)))
        .max_by(|a, b| a.0.cmp(&b.0))
        .map(|(_, s)| s);
    match best {
        Some(snap) => SnapshotSelection::Selected(snap.clone()),
        None => SnapshotSelection::OnlyPostClose,
    }
}

/// Read all snapshot files for one race from `snapshot_dir` (the per-race path,
/// used by the standalone loaders and tests). `trifecta` selects the filename
/// family (`_3t.json` vs exacta `.json`).
fn select_predeadline_snapshot(
    snapshot_dir: &Path,
    date: u32,
    stadium_code: u8,
    race_no: u8,
    trifecta: bool,
) -> SnapshotSelection {
    let prefix = format!("{}_{:02}_{:02}_", date, stadium_code, race_no);
    let entries = match fs::read_dir(snapshot_dir) {
        Ok(entries) => entries,
        Err(_) => return SnapshotSelection::NoSnapshots,
    };
    let snaps: Vec<SnapshotFile> = entries
        .flatten()
        .filter_map(|entry| {
            let name = entry.file_name().into_string().ok()?;
            if !name.starts_with(&prefix) || !name.ends_with(".json") {
                return None;
            }
            // Exacta and trifecta share the prefix; disambiguate by the _3t suffix.
            if name.ends_with("_3t.json") != trifecta {
                return None;
            }
            let content = fs::read_to_string(entry.path()).ok()?;
            serde_json::from_str::<SnapshotFile>(&content).ok()
        })
        .collect();
    choose_predeadline(&snaps)
}

/// Convert a raw selection into an exacta `PreDeadlineOdds`, parsing odds keys.
fn selection_to_exacta(selection: SnapshotSelection) -> PreDeadlineOdds<(u8, u8)> {
    match selection {
        SnapshotSelection::Selected(snap) => PreDeadlineOdds::Selected {
            odds: snap
                .odds
                .iter()
                .filter_map(|(k, &v)| parse_exacta_key(k).map(|key| (key, v)))
                .collect(),
            scraped_at: snap.scraped_at,
            deadline: snap.deadline.unwrap_or_default(),
            deadline_exact: snap.deadline_exact,
        },
        SnapshotSelection::OnlyPostClose => PreDeadlineOdds::OnlyPostClose,
        SnapshotSelection::NoSnapshots => PreDeadlineOdds::NoSnapshots,
    }
}

/// Convert a raw selection into a trifecta `PreDeadlineOdds`, parsing odds keys.
fn selection_to_trifecta(selection: SnapshotSelection) -> PreDeadlineOdds<(u8, u8, u8)> {
    match selection {
        SnapshotSelection::Selected(snap) => PreDeadlineOdds::Selected {
            odds: snap
                .odds
                .iter()
                .filter_map(|(k, &v)| parse_trifecta_key(k).map(|key| (key, v)))
                .collect(),
            scraped_at: snap.scraped_at,
            deadline: snap.deadline.unwrap_or_default(),
            deadline_exact: snap.deadline_exact,
        },
        SnapshotSelection::OnlyPostClose => PreDeadlineOdds::OnlyPostClose,
        SnapshotSelection::NoSnapshots => PreDeadlineOdds::NoSnapshots,
    }
}

/// Parse the race key from a snapshot filename `{date}_{stadium}_{race}_{ts}...`.
fn parse_snapshot_race_key(filename: &str) -> Option<(u32, u8, u8)> {
    let parts: Vec<&str> = filename.split('_').collect();
    if parts.len() < 4 {
        return None;
    }
    let date: u32 = parts[0].parse().ok()?;
    let stadium: u8 = parts[1].parse().ok()?;
    let race: u8 = parts[2].parse().ok()?;
    Some((date, stadium, race))
}

/// All snapshots under a directory, scanned once and grouped by race, so a
/// backtest can select per-race pre-deadline odds without re-reading the
/// directory for every race (which would be O(races × files)).
pub struct SnapshotIndex {
    /// Keyed by `(date, stadium_code, race_no, is_trifecta)`.
    by_race: HashMap<(u32, u8, u8, bool), Vec<SnapshotFile>>,
}

impl SnapshotIndex {
    /// Scan `snapshot_dir` once. A missing or unreadable directory yields an
    /// empty index (every race resolves to `NoSnapshots`).
    pub fn load<P: AsRef<Path>>(snapshot_dir: P) -> Self {
        let mut by_race: HashMap<(u32, u8, u8, bool), Vec<SnapshotFile>> = HashMap::new();
        if let Ok(entries) = fs::read_dir(snapshot_dir) {
            for entry in entries.flatten() {
                let Ok(name) = entry.file_name().into_string() else {
                    continue;
                };
                if !name.ends_with(".json") {
                    continue;
                }
                let is_trifecta = name.ends_with("_3t.json");
                let Some((date, stadium, race)) = parse_snapshot_race_key(&name) else {
                    continue;
                };
                let Ok(content) = fs::read_to_string(entry.path()) else {
                    continue;
                };
                let Ok(snap) = serde_json::from_str::<SnapshotFile>(&content) else {
                    continue;
                };
                by_race
                    .entry((date, stadium, race, is_trifecta))
                    .or_default()
                    .push(snap);
            }
        }
        Self { by_race }
    }

    fn select(&self, date: u32, stadium: u8, race: u8, trifecta: bool) -> SnapshotSelection {
        match self.by_race.get(&(date, stadium, race, trifecta)) {
            Some(snaps) => choose_predeadline(snaps),
            None => SnapshotSelection::NoSnapshots,
        }
    }

    /// Latest pre-deadline exacta odds for a race, lookahead-free.
    #[must_use]
    pub fn select_exacta(&self, date: u32, stadium: u8, race: u8) -> PreDeadlineOdds<(u8, u8)> {
        selection_to_exacta(self.select(date, stadium, race, false))
    }

    /// Latest pre-deadline trifecta odds for a race, lookahead-free.
    #[must_use]
    pub fn select_trifecta(
        &self,
        date: u32,
        stadium: u8,
        race: u8,
    ) -> PreDeadlineOdds<(u8, u8, u8)> {
        selection_to_trifecta(self.select(date, stadium, race, true))
    }
}

/// Load the latest pre-deadline exacta snapshot for a race, lookahead-free.
///
/// Returns the odds of the newest snapshot captured strictly before the race's
/// deadline, or a fail-closed variant when none is verifiably pre-deadline.
pub fn load_predeadline_exacta_odds<P: AsRef<Path>>(
    snapshot_dir: P,
    date: u32,
    stadium_code: u8,
    race_no: u8,
) -> PreDeadlineOdds<(u8, u8)> {
    selection_to_exacta(select_predeadline_snapshot(
        snapshot_dir.as_ref(),
        date,
        stadium_code,
        race_no,
        false,
    ))
}

/// Load the latest pre-deadline trifecta snapshot for a race, lookahead-free.
pub fn load_predeadline_trifecta_odds<P: AsRef<Path>>(
    snapshot_dir: P,
    date: u32,
    stadium_code: u8,
    race_no: u8,
) -> PreDeadlineOdds<(u8, u8, u8)> {
    selection_to_trifecta(select_predeadline_snapshot(
        snapshot_dir.as_ref(),
        date,
        stadium_code,
        race_no,
        true,
    ))
}

/// Parse exacta key "1-2" to (1, 2)
fn parse_exacta_key(key: &str) -> Option<(u8, u8)> {
    let parts: Vec<&str> = key.split('-').collect();
    if parts.len() != 2 {
        return None;
    }
    let first: u8 = parts[0].parse().ok()?;
    let second: u8 = parts[1].parse().ok()?;
    Some((first, second))
}

/// Parse trifecta key "1-2-3" to (1, 2, 3)
fn parse_trifecta_key(key: &str) -> Option<(u8, u8, u8)> {
    let parts: Vec<&str> = key.split('-').collect();
    if parts.len() != 3 {
        return None;
    }
    let first: u8 = parts[0].parse().ok()?;
    let second: u8 = parts[1].parse().ok()?;
    let third: u8 = parts[2].parse().ok()?;
    Some((first, second, third))
}

/// Check if exacta odds file exists
pub fn exacta_odds_exists<P: AsRef<Path>>(
    odds_dir: P,
    date: u32,
    stadium_code: u8,
    race_no: u8,
) -> bool {
    let filename = format!("{}_{:02}_{:02}.json", date, stadium_code, race_no);
    odds_dir.as_ref().join(&filename).exists()
}

/// Check if trifecta odds file exists
pub fn trifecta_odds_exists<P: AsRef<Path>>(
    odds_dir: P,
    date: u32,
    stadium_code: u8,
    race_no: u8,
) -> bool {
    let filename = format!("{}_{:02}_{:02}_3t.json", date, stadium_code, race_no);
    odds_dir.as_ref().join(&filename).exists()
}

/// List all available odds files in directory
pub fn list_odds_files<P: AsRef<Path>>(odds_dir: P) -> Vec<(u32, u8, u8, bool)> {
    let mut results = Vec::new();

    if let Ok(entries) = fs::read_dir(odds_dir) {
        for entry in entries.flatten() {
            if let Some(filename) = entry.file_name().to_str() {
                // Parse filename: {date}_{stadium:02}_{race:02}.json or _3t.json
                if filename.ends_with("_3t.json") {
                    // Trifecta file
                    let base = filename.trim_end_matches("_3t.json");
                    if let Some((date, stadium, race)) = parse_filename(base) {
                        results.push((date, stadium, race, true));
                    }
                } else if filename.ends_with(".json") && !filename.contains("_3t") {
                    // Exacta file
                    let base = filename.trim_end_matches(".json");
                    if let Some((date, stadium, race)) = parse_filename(base) {
                        results.push((date, stadium, race, false));
                    }
                }
            }
        }
    }

    results.sort();
    results
}

/// Parse filename "20240115_03_01" to (date, stadium, race)
fn parse_filename(base: &str) -> Option<(u32, u8, u8)> {
    let parts: Vec<&str> = base.split('_').collect();
    if parts.len() != 3 {
        return None;
    }
    let date: u32 = parts[0].parse().ok()?;
    let stadium: u8 = parts[1].parse().ok()?;
    let race: u8 = parts[2].parse().ok()?;
    Some((date, stadium, race))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_exacta_key() {
        assert_eq!(parse_exacta_key("1-2"), Some((1, 2)));
        assert_eq!(parse_exacta_key("6-5"), Some((6, 5)));
        assert_eq!(parse_exacta_key("invalid"), None);
        assert_eq!(parse_exacta_key("1-2-3"), None);
    }

    #[test]
    fn test_parse_trifecta_key() {
        assert_eq!(parse_trifecta_key("1-2-3"), Some((1, 2, 3)));
        assert_eq!(parse_trifecta_key("6-5-4"), Some((6, 5, 4)));
        assert_eq!(parse_trifecta_key("invalid"), None);
        assert_eq!(parse_trifecta_key("1-2"), None);
    }

    #[test]
    fn test_parse_filename() {
        assert_eq!(parse_filename("20240115_03_01"), Some((20240115, 3, 1)));
        assert_eq!(parse_filename("20231231_24_12"), Some((20231231, 24, 12)));
        assert_eq!(parse_filename("invalid"), None);
    }

    /// Write a snapshot JSON file into `dir`. `deadline` of `None` omits the key.
    fn write_snapshot(
        dir: &Path,
        filename: &str,
        date: u32,
        scraped_at: &str,
        deadline: Option<&str>,
        odds_key: &str,
    ) {
        let deadline_field = match deadline {
            Some(d) => format!("\"deadline\": \"{d}\",\n"),
            None => "\"deadline\": null,\n".to_string(),
        };
        let json = format!(
            "{{\n\"date\": {date},\n\"stadium_code\": 23,\n\"race_no\": 1,\n\
             \"scraped_at\": \"{scraped_at}\",\n{deadline_field}\
             \"deadline_exact\": true,\n\"bet_type\": \"exacta\",\n\
             \"odds\": {{ \"{odds_key}\": 5.6 }}\n}}"
        );
        fs::write(dir.join(filename), json).unwrap();
    }

    /// Fresh, empty temp dir unique to a test (cleaned first).
    fn fixture_dir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("boatrace_predeadline_{name}"));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn test_predeadline_selects_latest_before_deadline() {
        let dir = fixture_dir("latest");
        // deadline 15:24 JST = 06:24Z. Three pre-deadline captures + one post.
        write_snapshot(
            &dir,
            "20260620_23_01_a.json",
            20260620,
            "2026-06-20T05:24:00+00:00",
            Some("15:24"),
            "1-2",
        );
        write_snapshot(
            &dir,
            "20260620_23_01_b.json",
            20260620,
            "2026-06-20T06:14:00+00:00",
            Some("15:24"),
            "2-3",
        );
        write_snapshot(
            &dir,
            "20260620_23_01_c.json",
            20260620,
            "2026-06-20T06:23:00+00:00",
            Some("15:24"),
            "3-4",
        );
        write_snapshot(
            &dir,
            "20260620_23_01_d.json",
            20260620,
            "2026-06-20T06:30:00+00:00",
            Some("15:24"),
            "4-5",
        );

        let result = load_predeadline_exacta_odds(&dir, 20260620, 23, 1);
        match result {
            PreDeadlineOdds::Selected {
                odds,
                scraped_at,
                deadline_exact,
                ..
            } => {
                // The 06:23Z capture (latest still before 06:24Z) wins.
                assert!(scraped_at.starts_with("2026-06-20T06:23"));
                assert_eq!(odds.get(&(3, 4)), Some(&5.6));
                assert!(deadline_exact);
            }
            other => panic!("expected Selected, got {other:?}"),
        }
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_predeadline_timezone_straddle() {
        // The highest-risk bug: comparing JST deadline against UTC scraped_at.
        // deadline 15:24 JST == 06:24Z. 06:23Z is before; 06:25Z is after.
        // A naive (no +09:00) comparison would wrongly keep BOTH (both < 15:24
        // by wall clock) and pick the later 06:25Z — this test catches that.
        let dir = fixture_dir("tz");
        write_snapshot(
            &dir,
            "20260620_23_01_pre.json",
            20260620,
            "2026-06-20T06:23:00+00:00",
            Some("15:24"),
            "1-2",
        );
        write_snapshot(
            &dir,
            "20260620_23_01_post.json",
            20260620,
            "2026-06-20T06:25:00+00:00",
            Some("15:24"),
            "5-6",
        );

        match load_predeadline_exacta_odds(&dir, 20260620, 23, 1) {
            PreDeadlineOdds::Selected {
                scraped_at, odds, ..
            } => {
                assert!(
                    scraped_at.starts_with("2026-06-20T06:23"),
                    "must pick the pre-deadline capture, got {scraped_at}"
                );
                assert_eq!(odds.get(&(1, 2)), Some(&5.6));
            }
            other => panic!("expected Selected, got {other:?}"),
        }
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_predeadline_only_post_close() {
        let dir = fixture_dir("post");
        // 06:25Z is after the 06:24Z deadline -> excluded -> OnlyPostClose.
        write_snapshot(
            &dir,
            "20260620_23_01_a.json",
            20260620,
            "2026-06-20T06:25:00+00:00",
            Some("15:24"),
            "1-2",
        );
        assert_eq!(
            load_predeadline_exacta_odds(&dir, 20260620, 23, 1),
            PreDeadlineOdds::OnlyPostClose
        );
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_predeadline_no_snapshots() {
        let dir = fixture_dir("none");
        // Empty dir, and a different race present, both yield NoSnapshots.
        write_snapshot(
            &dir,
            "20260620_23_02_a.json",
            20260620,
            "2026-06-20T06:00:00+00:00",
            Some("15:24"),
            "1-2",
        );
        assert_eq!(
            load_predeadline_exacta_odds(&dir, 20260620, 23, 1),
            PreDeadlineOdds::NoSnapshots
        );
        // A non-existent dir is also NoSnapshots, not a panic.
        assert_eq!(
            load_predeadline_exacta_odds(dir.join("missing"), 20260620, 23, 1),
            PreDeadlineOdds::NoSnapshots
        );
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_predeadline_missing_deadline_excluded() {
        let dir = fixture_dir("nodeadline");
        // A snapshot without a deadline cannot prove it predates close -> excluded.
        write_snapshot(
            &dir,
            "20260620_23_01_a.json",
            20260620,
            "2026-06-20T06:00:00+00:00",
            None,
            "1-2",
        );
        assert_eq!(
            load_predeadline_exacta_odds(&dir, 20260620, 23, 1),
            PreDeadlineOdds::OnlyPostClose
        );
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_predeadline_naive_timestamp_excluded() {
        let dir = fixture_dir("naive");
        // Legacy naive timestamp (no offset) is unparseable as an instant ->
        // fail-closed exclusion rather than guessing its timezone.
        write_snapshot(
            &dir,
            "20260620_23_01_a.json",
            20260620,
            "2026-06-20T06:00:00.123456",
            Some("15:24"),
            "1-2",
        );
        assert_eq!(
            load_predeadline_exacta_odds(&dir, 20260620, 23, 1),
            PreDeadlineOdds::OnlyPostClose
        );
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_snapshot_index_matches_per_race_loader() {
        // The pre-indexed path must agree with the per-race directory loader.
        let dir = fixture_dir("index");
        write_snapshot(
            &dir,
            "20260620_23_01_a.json",
            20260620,
            "2026-06-20T06:14:00+00:00",
            Some("15:24"),
            "1-2",
        );
        write_snapshot(
            &dir,
            "20260620_23_01_b.json",
            20260620,
            "2026-06-20T06:23:00+00:00",
            Some("15:24"),
            "3-4",
        );
        write_snapshot(
            &dir,
            "20260620_23_01_c.json",
            20260620,
            "2026-06-20T06:30:00+00:00",
            Some("15:24"),
            "5-6",
        );
        write_snapshot(
            &dir,
            "20260620_23_02_a.json",
            20260620,
            "2026-06-20T06:50:00+00:00",
            Some("16:00"),
            "1-2",
        );

        let index = SnapshotIndex::load(&dir);
        // R1: the 06:23Z capture wins (latest before 06:24Z); index == per-race loader.
        assert_eq!(
            index.select_exacta(20260620, 23, 1),
            load_predeadline_exacta_odds(&dir, 20260620, 23, 1)
        );
        match index.select_exacta(20260620, 23, 1) {
            PreDeadlineOdds::Selected {
                scraped_at, odds, ..
            } => {
                assert!(scraped_at.starts_with("2026-06-20T06:23"));
                assert_eq!(odds.get(&(3, 4)), Some(&5.6));
            }
            other => panic!("expected Selected, got {other:?}"),
        }
        // A race absent from the index resolves to NoSnapshots.
        assert_eq!(
            index.select_exacta(20260620, 23, 9),
            PreDeadlineOdds::NoSnapshots
        );
        // A missing directory yields an empty index, not a panic.
        assert_eq!(
            SnapshotIndex::load(dir.join("missing")).select_exacta(20260620, 23, 1),
            PreDeadlineOdds::NoSnapshots
        );
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_predeadline_exacta_ignores_trifecta_files() {
        let dir = fixture_dir("suffix");
        // A co-located trifecta snapshot must not satisfy an exacta query, and
        // vice versa.
        write_snapshot(
            &dir,
            "20260620_23_01_a_3t.json",
            20260620,
            "2026-06-20T06:00:00+00:00",
            Some("15:24"),
            "1-2-3",
        );
        assert_eq!(
            load_predeadline_exacta_odds(&dir, 20260620, 23, 1),
            PreDeadlineOdds::NoSnapshots
        );

        match load_predeadline_trifecta_odds(&dir, 20260620, 23, 1) {
            PreDeadlineOdds::Selected { odds, .. } => assert_eq!(odds.get(&(1, 2, 3)), Some(&5.6)),
            other => panic!("expected trifecta Selected, got {other:?}"),
        }
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_load_exacta_from_json_string() {
        let json = r#"{
            "date": 20240115,
            "stadium_code": 3,
            "race_no": 1,
            "scraped_at": "2025-12-30T21:03:54.919741",
            "exacta": {
                "1-2": 7.6,
                "2-1": 20.3
            }
        }"#;

        let odds_file: ExactaOddsFile = serde_json::from_str(json).unwrap();
        assert_eq!(odds_file.date, 20240115);
        assert_eq!(odds_file.stadium_code, 3);
        assert_eq!(odds_file.race_no, 1);
        assert_eq!(odds_file.exacta.len(), 2);
        assert!((odds_file.exacta["1-2"] - 7.6).abs() < 0.01);
    }
}
