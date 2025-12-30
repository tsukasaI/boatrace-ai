//! Raw data parser for boatrace text files
//!
//! Parses fixed-width CP932-encoded text files from boatrace.jp
//!
//! # Example
//!
//! ```no_run
//! use boatrace::data::parser::{ProgramParser, ResultParser};
//! use std::path::Path;
//!
//! let parser = ProgramParser::new();
//! for (race, entries) in parser.parse_file(Path::new("programs_20240115.txt")).unwrap() {
//!     println!("Race {} at {}: {} entries", race.race_no, race.stadium_name, entries.len());
//! }
//! ```

use encoding_rs::SHIFT_JIS;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::Path;

/// Race information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaceInfo {
    pub date: String,
    pub stadium_code: u8,
    pub stadium_name: String,
    pub race_no: u8,
    pub race_type: String,
    pub distance: u16,
}

/// Racer entry from program file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedRacerEntry {
    pub boat_no: u8,
    pub racer_id: u32,
    pub racer_name: String,
    pub age: u8,
    pub branch: String,
    pub weight: u8,
    pub racer_class: String,
    pub national_win_rate: f64,
    pub national_in2_rate: f64,
    pub local_win_rate: f64,
    pub local_in2_rate: f64,
    pub motor_no: u16,
    pub motor_in2_rate: f64,
    pub boat_no_equip: u16,
    pub boat_in2_rate: f64,
}

/// Race result entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaceResult {
    pub boat_no: u8,
    pub racer_id: u32,
    pub rank: u8,
    pub race_time: String,
    pub course: u8,
    pub start_timing: f64,
}

/// Race payouts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RacePayouts {
    pub date: String,
    pub stadium_code: u8,
    pub race_no: u8,
    pub win: HashMap<u8, u32>,
    pub place: HashMap<u8, u32>,
    pub exacta: HashMap<(u8, u8), u32>,
    pub quinella: HashMap<(u8, u8), u32>,
    pub wide: HashMap<(u8, u8), u32>,
    pub trifecta: HashMap<(u8, u8, u8), u32>,
    pub trio: HashMap<(u8, u8, u8), u32>,
}

impl RacePayouts {
    fn new(date: String, stadium_code: u8, race_no: u8) -> Self {
        Self {
            date,
            stadium_code,
            race_no,
            win: HashMap::new(),
            place: HashMap::new(),
            exacta: HashMap::new(),
            quinella: HashMap::new(),
            wide: HashMap::new(),
            trifecta: HashMap::new(),
            trio: HashMap::new(),
        }
    }

    fn has_payouts(&self) -> bool {
        !self.win.is_empty()
            || !self.place.is_empty()
            || !self.exacta.is_empty()
            || !self.trifecta.is_empty()
    }
}

/// Stadium code to name mapping
fn stadium_name(code: u8) -> &'static str {
    match code {
        1 => "桐生",
        2 => "戸田",
        3 => "江戸川",
        4 => "平和島",
        5 => "多摩川",
        6 => "浜名湖",
        7 => "蒲郡",
        8 => "常滑",
        9 => "津",
        10 => "三国",
        11 => "びわこ",
        12 => "住之江",
        13 => "尼崎",
        14 => "鳴門",
        15 => "丸亀",
        16 => "児島",
        17 => "宮島",
        18 => "徳山",
        19 => "下関",
        20 => "若松",
        21 => "芦屋",
        22 => "福岡",
        23 => "唐津",
        24 => "大村",
        _ => "不明",
    }
}

/// Convert fullwidth digits to halfwidth
fn normalize_fullwidth_numbers(text: &str) -> String {
    let fullwidth = "０１２３４５６７８９";
    let halfwidth = "0123456789";

    text.chars()
        .map(|c| {
            if let Some(idx) = fullwidth.find(c) {
                halfwidth.chars().nth(idx / 3).unwrap_or(c)
            } else {
                c
            }
        })
        .collect()
}

/// Read file with CP932 encoding
fn read_cp932_file(path: &Path) -> io::Result<String> {
    let bytes = fs::read(path)?;

    // Try SHIFT_JIS (CP932) first
    let (decoded, _, had_errors) = SHIFT_JIS.decode(&bytes);

    if had_errors {
        // Fallback to UTF-8
        String::from_utf8(bytes)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))
    } else {
        Ok(decoded.into_owned())
    }
}

/// Program file parser
pub struct ProgramParser {
    stadium_pattern: Regex,
    race_pattern: Regex,
    racer_pattern: Regex,
}

impl Default for ProgramParser {
    fn default() -> Self {
        Self::new()
    }
}

impl ProgramParser {
    pub fn new() -> Self {
        Self {
            stadium_pattern: Regex::new(r"^(\d{2})BBGN").unwrap(),
            race_pattern: Regex::new(r"[　\s]*([０-９\d]+)Ｒ\s*(.*?)\s*Ｈ(\d+)").unwrap(),
            racer_pattern: Regex::new(
                r"^(\d)\s+(\d{4})(.{4})(\d{2})(.{2})(\d{2})([AB][12])\s*(\d+\.\d+)\s*(\d+\.\d+)\s*(\d+\.\d+)\s*(\d+\.\d+)\s*(\d+)\s*(\d+\.\d+)\s*(\d+)\s*(\d+\.\d+)",
            ).unwrap(),
        }
    }

    /// Parse a program file
    pub fn parse_file(&self, path: &Path) -> io::Result<Vec<(RaceInfo, Vec<ParsedRacerEntry>)>> {
        let content = read_cp932_file(path)?;
        let lines: Vec<&str> = content.lines().collect();

        // Extract date from filename
        let date_str = path
            .file_stem()
            .and_then(|s| s.to_str())
            .and_then(|s| s.split('_').next_back())
            .unwrap_or("unknown")
            .to_string();

        let mut results = Vec::new();
        let mut current_stadium: Option<u8> = None;
        let mut current_race: Option<RaceInfo> = None;
        let mut current_racers: Vec<ParsedRacerEntry> = Vec::new();

        for line in lines {
            // Stadium identification
            if let Some(caps) = self.stadium_pattern.captures(line) {
                if let Ok(code) = caps[1].parse::<u8>() {
                    current_stadium = Some(code);
                }
                continue;
            }

            // Race information
            if let Some(caps) = self.race_pattern.captures(line) {
                if let Some(stadium_code) = current_stadium {
                    // Save previous race
                    if let Some(race) = current_race.take() {
                        if !current_racers.is_empty() {
                            results.push((race, std::mem::take(&mut current_racers)));
                        }
                    }

                    let race_no_str = normalize_fullwidth_numbers(&caps[1]);
                    let race_no: u8 = race_no_str.parse().unwrap_or(0);
                    let race_type = caps[2].trim().to_string();
                    let distance: u16 = caps[3].parse().unwrap_or(1800);

                    current_race = Some(RaceInfo {
                        date: date_str.clone(),
                        stadium_code,
                        stadium_name: stadium_name(stadium_code).to_string(),
                        race_no,
                        race_type,
                        distance,
                    });
                }
                continue;
            }

            // Racer entry
            if current_race.is_some() && !line.trim().is_empty() {
                if let Some(first_char) = line.chars().next() {
                    if first_char.is_ascii_digit() {
                        if let Some(racer) = self.parse_racer_line(line) {
                            current_racers.push(racer);
                        }
                    }
                }
            }
        }

        // Save last race
        if let Some(race) = current_race {
            if !current_racers.is_empty() {
                results.push((race, current_racers));
            }
        }

        Ok(results)
    }

    fn parse_racer_line(&self, line: &str) -> Option<ParsedRacerEntry> {
        // Try regex first
        if let Some(caps) = self.racer_pattern.captures(line) {
            return Some(ParsedRacerEntry {
                boat_no: caps[1].parse().ok()?,
                racer_id: caps[2].parse().ok()?,
                racer_name: caps[3].trim().to_string(),
                age: caps[4].parse().ok()?,
                branch: caps[5].trim().to_string(),
                weight: caps[6].parse().ok()?,
                racer_class: caps[7].to_string(),
                national_win_rate: caps[8].parse().ok()?,
                national_in2_rate: caps[9].parse().ok()?,
                local_win_rate: caps[10].parse().ok()?,
                local_in2_rate: caps[11].parse().ok()?,
                motor_no: caps[12].parse().ok()?,
                motor_in2_rate: caps[13].parse().ok()?,
                boat_no_equip: caps[14].parse().ok()?,
                boat_in2_rate: caps[15].parse().ok()?,
            });
        }

        // Fallback: fixed position parsing
        if line.len() < 20 {
            return None;
        }

        let boat_no: u8 = line.get(0..1)?.parse().ok()?;
        let racer_id: u32 = line.get(2..6)?.parse().ok()?;
        let racer_name = line.get(6..10)?.trim().to_string();
        let age: u8 = line.get(10..12)?.parse().ok()?;
        let branch = line.get(12..14)?.trim().to_string();
        let weight: u8 = line.get(14..16)?.parse().ok()?;
        let racer_class = line.get(16..18)?.to_string();

        let remaining: Vec<&str> = line.get(18..)?.split_whitespace().collect();
        if remaining.len() < 8 {
            return None;
        }

        Some(ParsedRacerEntry {
            boat_no,
            racer_id,
            racer_name,
            age,
            branch,
            weight,
            racer_class,
            national_win_rate: remaining[0].parse().ok()?,
            national_in2_rate: remaining[1].parse().ok()?,
            local_win_rate: remaining[2].parse().ok()?,
            local_in2_rate: remaining[3].parse().ok()?,
            motor_no: remaining[4].parse().ok()?,
            motor_in2_rate: remaining[5].parse().ok()?,
            boat_no_equip: remaining[6].parse().ok()?,
            boat_in2_rate: remaining[7].parse().ok()?,
        })
    }
}

/// Result file parser
pub struct ResultParser {
    stadium_pattern: Regex,
    race_pattern: Regex,
    result_pattern: Regex,
}

impl Default for ResultParser {
    fn default() -> Self {
        Self::new()
    }
}

impl ResultParser {
    pub fn new() -> Self {
        Self {
            stadium_pattern: Regex::new(r"^(\d{2})KBGN").unwrap(),
            race_pattern: Regex::new(r"^\s*(\d+)R\s+(.*?)\s+H(\d+)").unwrap(),
            result_pattern: Regex::new(r"^\s*(\d{2})\s+(\d)\s+(\d{4})").unwrap(),
        }
    }

    /// Parse a result file
    pub fn parse_file(&self, path: &Path) -> io::Result<Vec<(RaceInfo, Vec<RaceResult>)>> {
        let content = read_cp932_file(path)?;
        let lines: Vec<&str> = content.lines().collect();

        let date_str = path
            .file_stem()
            .and_then(|s| s.to_str())
            .and_then(|s| s.split('_').next_back())
            .unwrap_or("unknown")
            .to_string();

        let mut results = Vec::new();
        let mut current_stadium: Option<u8> = None;
        let mut current_race: Option<RaceInfo> = None;
        let mut current_results: Vec<RaceResult> = Vec::new();
        let mut in_results_section = false;

        for line in lines {
            // Stadium identification
            if let Some(caps) = self.stadium_pattern.captures(line) {
                if let Ok(code) = caps[1].parse::<u8>() {
                    current_stadium = Some(code);
                }
                continue;
            }

            // Race information
            if let Some(caps) = self.race_pattern.captures(line) {
                if let Some(stadium_code) = current_stadium {
                    // Save previous race
                    if let Some(race) = current_race.take() {
                        if !current_results.is_empty() {
                            results.push((race, std::mem::take(&mut current_results)));
                        }
                    }

                    let race_no_str = normalize_fullwidth_numbers(&caps[1]);
                    let race_no: u8 = race_no_str.parse().unwrap_or(0);
                    let race_type = caps[2].trim().to_string();
                    let distance: u16 = caps[3].parse().unwrap_or(1800);

                    current_race = Some(RaceInfo {
                        date: date_str.clone(),
                        stadium_code,
                        stadium_name: stadium_name(stadium_code).to_string(),
                        race_no,
                        race_type,
                        distance,
                    });
                    in_results_section = false;
                }
                continue;
            }

            // Results section starts at separator
            if line.starts_with("---") && current_race.is_some() {
                in_results_section = true;
                continue;
            }

            // Parse result lines
            if in_results_section && current_race.is_some() && !line.trim().is_empty() {
                if let Some(result) = self.parse_result_line(line) {
                    current_results.push(result);
                } else if line.contains("単勝") || line.contains("複勝") {
                    in_results_section = false;
                }
            }
        }

        // Save last race
        if let Some(race) = current_race {
            if !current_results.is_empty() {
                results.push((race, current_results));
            }
        }

        Ok(results)
    }

    fn parse_result_line(&self, line: &str) -> Option<RaceResult> {
        let caps = self.result_pattern.captures(line)?;

        let rank: u8 = caps[1].parse().ok()?;
        let boat_no: u8 = caps[2].parse().ok()?;
        let racer_id: u32 = caps[3].parse().ok()?;

        let remaining: Vec<&str> = line[caps.get(0)?.end()..].split_whitespace().collect();

        let mut course: u8 = 0;
        let mut start_timing: f64 = 0.0;
        let mut race_time = String::new();

        if remaining.len() >= 3 {
            race_time = remaining[remaining.len() - 1].to_string();
            start_timing = remaining[remaining.len() - 2].parse().unwrap_or(0.0);
            course = remaining[remaining.len() - 3].parse().unwrap_or(0);
        }

        Some(RaceResult {
            boat_no,
            racer_id,
            rank,
            race_time,
            course,
            start_timing,
        })
    }
}

/// Payout file parser
pub struct PayoutParser {
    stadium_pattern: Regex,
    race_pattern: Regex,
    win_pattern: Regex,
    place_pattern: Regex,
    exacta_pattern: Regex,
    quinella_pattern: Regex,
    wide_pattern: Regex,
    trifecta_pattern: Regex,
    trio_pattern: Regex,
}

impl Default for PayoutParser {
    fn default() -> Self {
        Self::new()
    }
}

impl PayoutParser {
    pub fn new() -> Self {
        Self {
            stadium_pattern: Regex::new(r"^(\d{2})KBGN").unwrap(),
            race_pattern: Regex::new(r"^\s*(\d+)R\s+").unwrap(),
            win_pattern: Regex::new(r"単勝\s+(\d)\s+(\d+)").unwrap(),
            place_pattern: Regex::new(r"複勝\s+(\d)\s+(\d+)").unwrap(),
            exacta_pattern: Regex::new(r"２連単\s+(\d)-(\d)\s+(\d+)").unwrap(),
            quinella_pattern: Regex::new(r"２連複\s+(\d)-(\d)\s+(\d+)").unwrap(),
            wide_pattern: Regex::new(r"(?:ワイド|ﾜｲﾄﾞ)\s+(\d)-(\d)\s+(\d+)").unwrap(),
            trifecta_pattern: Regex::new(r"３連単\s+(\d)-(\d)-(\d)\s+(\d+)").unwrap(),
            trio_pattern: Regex::new(r"３連複\s+(\d)-(\d)-(\d)\s+(\d+)").unwrap(),
        }
    }

    /// Parse a result file for payouts
    pub fn parse_file(&self, path: &Path) -> io::Result<Vec<RacePayouts>> {
        let content = read_cp932_file(path)?;
        let lines: Vec<&str> = content.lines().collect();

        let date_str = path
            .file_stem()
            .and_then(|s| s.to_str())
            .and_then(|s| s.split('_').next_back())
            .unwrap_or("unknown")
            .to_string();

        let mut results = Vec::new();
        let mut current_stadium: Option<u8> = None;
        let mut current_payouts: Option<RacePayouts> = None;
        let mut in_payout_section = false;

        for line in lines {
            // Stadium identification
            if let Some(caps) = self.stadium_pattern.captures(line) {
                if let Ok(code) = caps[1].parse::<u8>() {
                    current_stadium = Some(code);
                }
                continue;
            }

            // Race number
            if let Some(caps) = self.race_pattern.captures(line) {
                if let Some(stadium_code) = current_stadium {
                    // Save previous race
                    if let Some(payouts) = current_payouts.take() {
                        if payouts.has_payouts() {
                            results.push(payouts);
                        }
                    }

                    let race_no: u8 = caps[1].parse().unwrap_or(0);
                    current_payouts =
                        Some(RacePayouts::new(date_str.clone(), stadium_code, race_no));
                    in_payout_section = false;
                }
                continue;
            }

            // Detect payout section
            if line.contains("単勝") || line.contains("２連単") || line.contains("３連単") {
                in_payout_section = true;
            }

            // Parse payouts
            if in_payout_section {
                if let Some(ref mut payouts) = current_payouts {
                    self.parse_payout_line(line, payouts);
                }
            }
        }

        // Save last race
        if let Some(payouts) = current_payouts {
            if payouts.has_payouts() {
                results.push(payouts);
            }
        }

        Ok(results)
    }

    fn parse_payout_line(&self, line: &str, payouts: &mut RacePayouts) {
        // Win
        for caps in self.win_pattern.captures_iter(line) {
            if let (Ok(boat), Ok(payout)) = (caps[1].parse::<u8>(), caps[2].parse::<u32>()) {
                payouts.win.insert(boat, payout);
            }
        }

        // Place
        for caps in self.place_pattern.captures_iter(line) {
            if let (Ok(boat), Ok(payout)) = (caps[1].parse::<u8>(), caps[2].parse::<u32>()) {
                payouts.place.insert(boat, payout);
            }
        }

        // Exacta
        for caps in self.exacta_pattern.captures_iter(line) {
            if let (Ok(first), Ok(second), Ok(payout)) = (
                caps[1].parse::<u8>(),
                caps[2].parse::<u8>(),
                caps[3].parse::<u32>(),
            ) {
                payouts.exacta.insert((first, second), payout);
            }
        }

        // Quinella
        for caps in self.quinella_pattern.captures_iter(line) {
            if let (Ok(a), Ok(b), Ok(payout)) = (
                caps[1].parse::<u8>(),
                caps[2].parse::<u8>(),
                caps[3].parse::<u32>(),
            ) {
                let key = if a < b { (a, b) } else { (b, a) };
                payouts.quinella.insert(key, payout);
            }
        }

        // Wide
        for caps in self.wide_pattern.captures_iter(line) {
            if let (Ok(a), Ok(b), Ok(payout)) = (
                caps[1].parse::<u8>(),
                caps[2].parse::<u8>(),
                caps[3].parse::<u32>(),
            ) {
                let key = if a < b { (a, b) } else { (b, a) };
                payouts.wide.insert(key, payout);
            }
        }

        // Trifecta
        for caps in self.trifecta_pattern.captures_iter(line) {
            if let (Ok(first), Ok(second), Ok(third), Ok(payout)) = (
                caps[1].parse::<u8>(),
                caps[2].parse::<u8>(),
                caps[3].parse::<u8>(),
                caps[4].parse::<u32>(),
            ) {
                payouts.trifecta.insert((first, second, third), payout);
            }
        }

        // Trio
        for caps in self.trio_pattern.captures_iter(line) {
            if let (Ok(a), Ok(b), Ok(c), Ok(payout)) = (
                caps[1].parse::<u8>(),
                caps[2].parse::<u8>(),
                caps[3].parse::<u8>(),
                caps[4].parse::<u32>(),
            ) {
                let mut sorted = [a, b, c];
                sorted.sort();
                payouts
                    .trio
                    .insert((sorted[0], sorted[1], sorted[2]), payout);
            }
        }
    }
}

/// Flattened payout record for CSV export
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PayoutRecord {
    pub date: String,
    pub stadium_code: u8,
    pub race_no: u8,
    pub bet_type: String,
    pub first: u8,
    pub second: u8,
    pub third: u8,
    pub payout: u32,
    pub odds: f64,
}

/// Convert payouts to flattened records
pub fn flatten_payouts(payouts: &RacePayouts) -> Vec<PayoutRecord> {
    let mut records = Vec::new();

    // Exacta
    for (&(first, second), &payout) in &payouts.exacta {
        records.push(PayoutRecord {
            date: payouts.date.clone(),
            stadium_code: payouts.stadium_code,
            race_no: payouts.race_no,
            bet_type: "exacta".to_string(),
            first,
            second,
            third: 0,
            payout,
            odds: payout as f64 / 100.0,
        });
    }

    // Trifecta
    for (&(first, second, third), &payout) in &payouts.trifecta {
        records.push(PayoutRecord {
            date: payouts.date.clone(),
            stadium_code: payouts.stadium_code,
            race_no: payouts.race_no,
            bet_type: "trifecta".to_string(),
            first,
            second,
            third,
            payout,
            odds: payout as f64 / 100.0,
        });
    }

    records
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_fullwidth_numbers() {
        assert_eq!(normalize_fullwidth_numbers("１２３"), "123");
        assert_eq!(normalize_fullwidth_numbers("test０９"), "test09");
        assert_eq!(normalize_fullwidth_numbers("abc"), "abc");
    }

    #[test]
    fn test_stadium_name() {
        assert_eq!(stadium_name(1), "桐生");
        assert_eq!(stadium_name(23), "唐津");
        assert_eq!(stadium_name(24), "大村");
        assert_eq!(stadium_name(99), "不明");
    }

    #[test]
    fn test_race_payouts_new() {
        let payouts = RacePayouts::new("20240115".to_string(), 23, 1);
        assert_eq!(payouts.date, "20240115");
        assert_eq!(payouts.stadium_code, 23);
        assert_eq!(payouts.race_no, 1);
        assert!(!payouts.has_payouts());
    }

    #[test]
    fn test_race_payouts_has_payouts() {
        let mut payouts = RacePayouts::new("20240115".to_string(), 23, 1);
        assert!(!payouts.has_payouts());

        payouts.exacta.insert((1, 2), 500);
        assert!(payouts.has_payouts());
    }

    #[test]
    fn test_flatten_payouts() {
        let mut payouts = RacePayouts::new("20240115".to_string(), 23, 1);
        payouts.exacta.insert((1, 2), 500);
        payouts.trifecta.insert((1, 2, 3), 2500);

        let records = flatten_payouts(&payouts);
        assert_eq!(records.len(), 2);

        let exacta_record = records.iter().find(|r| r.bet_type == "exacta").unwrap();
        assert_eq!(exacta_record.first, 1);
        assert_eq!(exacta_record.second, 2);
        assert_eq!(exacta_record.payout, 500);
        assert!((exacta_record.odds - 5.0).abs() < 0.01);

        let trifecta_record = records.iter().find(|r| r.bet_type == "trifecta").unwrap();
        assert_eq!(trifecta_record.first, 1);
        assert_eq!(trifecta_record.second, 2);
        assert_eq!(trifecta_record.third, 3);
        assert!((trifecta_record.odds - 25.0).abs() < 0.01);
    }

    #[test]
    fn test_program_parser_new() {
        let parser = ProgramParser::new();
        assert!(parser.stadium_pattern.is_match("22BBGN"));
        assert!(!parser.stadium_pattern.is_match("22KBGN"));
    }

    #[test]
    fn test_result_parser_new() {
        let parser = ResultParser::new();
        assert!(parser.stadium_pattern.is_match("22KBGN"));
        assert!(!parser.stadium_pattern.is_match("22BBGN"));
    }

    #[test]
    fn test_payout_parser_patterns() {
        let parser = PayoutParser::new();

        // Test exacta pattern
        let line = "２連単  1-2  500";
        assert!(parser.exacta_pattern.is_match(line));

        // Test trifecta pattern
        let line = "３連単  1-2-3  2500";
        assert!(parser.trifecta_pattern.is_match(line));

        // Test win pattern
        let line = "単勝  1  150";
        assert!(parser.win_pattern.is_match(line));
    }

    #[test]
    fn test_payout_parser_parse_line() {
        let parser = PayoutParser::new();
        let mut payouts = RacePayouts::new("20240115".to_string(), 23, 1);

        parser.parse_payout_line("２連単  1-2  500", &mut payouts);
        assert_eq!(payouts.exacta.get(&(1, 2)), Some(&500));

        parser.parse_payout_line("３連単  1-2-3  2500", &mut payouts);
        assert_eq!(payouts.trifecta.get(&(1, 2, 3)), Some(&2500));

        parser.parse_payout_line("単勝  1  150", &mut payouts);
        assert_eq!(payouts.win.get(&1), Some(&150));
    }
}
