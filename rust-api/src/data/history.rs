//! Racer History Index
//!
//! Provides O(1) lookup of racer's past race results for computing historical features.

use polars::prelude::*;
use std::collections::HashMap;
use std::path::Path;

use crate::data::features::HistoricalFeatures;

/// Number of recent races to consider for historical features
const RECENT_RACE_LIMIT: usize = 30;

/// Single historical race entry for a racer
#[derive(Debug, Clone)]
pub struct HistoricalRaceEntry {
    pub date: u32,
    pub stadium_code: u8,
    pub race_no: u8,
    pub course: u8,       // Starting lane 1-6
    pub rank: u8,         // Finishing position 1-6
    pub start_timing: f64,
}

/// Historical data indexed by racer_id for O(1) lookup
pub struct RacerHistoryIndex {
    /// racer_id -> Vec of past race results, sorted by date descending
    history: HashMap<u32, Vec<HistoricalRaceEntry>>,
}

impl RacerHistoryIndex {
    /// Load and index all race results from CSV
    pub fn load<P: AsRef<Path>>(csv_path: P) -> Result<Self, PolarsError> {
        let df = CsvReadOptions::default()
            .try_into_reader_with_file_path(Some(csv_path.as_ref().to_path_buf()))?
            .finish()?;

        let date_col = df.column("date")?.i64()?;
        let stadium_col = df.column("stadium_code")?.i64()?;
        let race_col = df.column("race_no")?.i64()?;
        let racer_id_col = df.column("racer_id")?.i64()?;
        let rank_col = df.column("rank")?.i64()?;
        let course_col = df.column("course")?.i64()?;
        let st_col = df.column("start_timing")?.f64()?;

        let mut history: HashMap<u32, Vec<HistoricalRaceEntry>> = HashMap::new();

        for i in 0..df.height() {
            if let (Some(date), Some(stadium), Some(race), Some(racer_id), Some(rank), Some(course)) = (
                date_col.get(i),
                stadium_col.get(i),
                race_col.get(i),
                racer_id_col.get(i),
                rank_col.get(i),
                course_col.get(i),
            ) {
                // Skip invalid ranks (disqualified, etc.)
                if !(1..=6).contains(&rank) {
                    continue;
                }

                let result = HistoricalRaceEntry {
                    date: date as u32,
                    stadium_code: stadium as u8,
                    race_no: race as u8,
                    course: course as u8,
                    rank: rank as u8,
                    start_timing: st_col.get(i).unwrap_or(0.15),
                };

                history
                    .entry(racer_id as u32)
                    .or_default()
                    .push(result);
            }
        }

        // Sort each racer's history by date descending (most recent first)
        for results in history.values_mut() {
            results.sort_by(|a, b| b.date.cmp(&a.date));
        }

        Ok(Self { history })
    }

    /// Get recent race results for a racer before a given date
    pub fn get_recent_races(&self, racer_id: u32, before_date: u32, limit: usize) -> Vec<&HistoricalRaceEntry> {
        self.history
            .get(&racer_id)
            .map(|results| {
                results
                    .iter()
                    .filter(|r| r.date < before_date)
                    .take(limit)
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Compute historical features for a racer
    ///
    /// # Arguments
    /// * `racer_id` - The racer's ID
    /// * `stadium_code` - Current stadium (for local features)
    /// * `course` - Current course/lane (for course-specific features)
    /// * `boat_no` - Current boat number (for course-taking features)
    /// * `before_date` - Only consider races before this date
    pub fn compute_historical_features(
        &self,
        racer_id: u32,
        stadium_code: u8,
        course: u8,
        _boat_no: u8, // Reserved for future course-taking features
        before_date: u32,
    ) -> HistoricalFeatures {
        let recent_races = self.get_recent_races(racer_id, before_date, RECENT_RACE_LIMIT);

        if recent_races.is_empty() {
            return HistoricalFeatures::default();
        }

        // Overall recent stats
        let race_count = recent_races.len() as f64;
        let wins = recent_races.iter().filter(|r| r.rank == 1).count() as f64;
        let in2 = recent_races.iter().filter(|r| r.rank <= 2).count() as f64;
        let in3 = recent_races.iter().filter(|r| r.rank <= 3).count() as f64;
        let avg_rank: f64 = recent_races.iter().map(|r| r.rank as f64).sum::<f64>() / race_count;
        let avg_st: f64 = recent_races.iter().map(|r| r.start_timing).sum::<f64>() / race_count;

        // Start timing statistics
        let st_values: Vec<f64> = recent_races.iter().map(|r| r.start_timing).collect();
        let st_consistency = if st_values.len() > 1 {
            Self::std_dev(&st_values)
        } else {
            0.05 // Default
        };
        let flying_start_rate = st_values.iter().filter(|&&st| st < 0.0).count() as f64 / race_count;
        let late_start_rate = st_values.iter().filter(|&&st| st > 0.20).count() as f64 / race_count;

        // Local (stadium-specific) stats
        let local_races: Vec<_> = recent_races
            .iter()
            .filter(|r| r.stadium_code == stadium_code)
            .collect();
        let local_race_count = local_races.len() as f64;
        let local_wins = local_races.iter().filter(|r| r.rank == 1).count() as f64;
        let local_recent_win_rate = if local_race_count > 0.0 {
            local_wins / local_race_count
        } else {
            0.0
        };

        // Course-specific stats (same lane position)
        let course_races: Vec<_> = recent_races
            .iter()
            .filter(|r| r.course == course)
            .collect();
        let course_race_count = course_races.len() as f64;
        let course_wins = course_races.iter().filter(|r| r.rank == 1).count() as f64;
        let course_in2 = course_races.iter().filter(|r| r.rank <= 2).count() as f64;
        let course_win_rate = if course_race_count > 0.0 {
            course_wins / course_race_count
        } else {
            // Default to historical course advantage if no data
            Self::default_course_win_rate(course)
        };
        let course_in2_rate = if course_race_count > 0.0 {
            course_in2 / course_race_count
        } else {
            0.0
        };

        // Course-taking features (difference between actual course and boat_no)
        // Note: We use historical boat_no which we don't have, so approximate with course
        // This is computed differently in Python where we have boat_no from program
        let avg_course_diff = 0.0; // Default - computed differently in practice
        let inside_take_rate = 0.0; // Default - computed differently in practice

        // Weighted recent win rate (last 5 races)
        let recent_5: Vec<_> = recent_races.iter().take(5).collect();
        let weighted_recent_win = if recent_5.len() >= 5 {
            recent_5.iter().filter(|r| r.rank == 1).count() as f64 / recent_5.len() as f64
        } else {
            wins / race_count
        };

        HistoricalFeatures {
            recent_win_rate: wins / race_count,
            recent_in2_rate: in2 / race_count,
            recent_in3_rate: in3 / race_count,
            recent_avg_rank: avg_rank,
            recent_avg_st: avg_st,
            recent_race_count: race_count,
            local_recent_win_rate,
            local_race_count,
            course_win_rate,
            course_in2_rate,
            st_consistency,
            flying_start_rate,
            late_start_rate,
            avg_course_diff,
            inside_take_rate,
            weighted_recent_win,
        }
    }

    /// Calculate standard deviation
    fn std_dev(values: &[f64]) -> f64 {
        if values.len() <= 1 {
            return 0.0;
        }
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
        variance.sqrt()
    }

    /// Default course win rate based on historical averages
    fn default_course_win_rate(course: u8) -> f64 {
        match course {
            1 => 0.55,
            2 => 0.14,
            3 => 0.12,
            4 => 0.10,
            5 => 0.06,
            6 => 0.03,
            _ => 0.0,
        }
    }

    /// Number of unique racers in the index
    pub fn len(&self) -> usize {
        self.history.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.history.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_course_win_rate() {
        assert!((RacerHistoryIndex::default_course_win_rate(1) - 0.55).abs() < 0.01);
        assert!((RacerHistoryIndex::default_course_win_rate(6) - 0.03).abs() < 0.01);
    }

    #[test]
    fn test_empty_history() {
        let index = RacerHistoryIndex {
            history: HashMap::new(),
        };
        let features = index.compute_historical_features(9999, 1, 1, 1, 20240115);

        // Should return defaults
        assert_eq!(features.recent_race_count, 0.0);
        assert_eq!(features.recent_win_rate, 0.0);
    }
}
