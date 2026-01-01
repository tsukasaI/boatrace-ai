//! CSV data loading for race programs and entries

use polars::prelude::*;
use std::collections::HashMap;
use std::path::Path;

use crate::models::RacerEntry;

/// Program entry data loaded from CSV
#[derive(Debug, Clone)]
pub struct ProgramEntry {
    pub date: u32,
    pub stadium_code: u8,
    pub race_no: u8,
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
    pub motor_no: u32,
    pub motor_in2_rate: f64,
    pub boat_no_equip: u32,
    pub boat_in2_rate: f64,
}

impl ProgramEntry {
    /// Convert to RacerEntry for prediction
    pub fn to_racer_entry(&self) -> RacerEntry {
        RacerEntry {
            boat_no: self.boat_no,
            racer_id: self.racer_id,
            racer_name: self.racer_name.clone(),
            age: self.age,
            weight: self.weight,
            racer_class: self.racer_class.clone(),
            branch: self.branch.clone(),
            national_win_rate: self.national_win_rate,
            national_in2_rate: self.national_in2_rate,
            local_win_rate: self.local_win_rate,
            local_in2_rate: self.local_in2_rate,
            motor_no: self.motor_no,
            motor_in2_rate: self.motor_in2_rate,
            boat_no_equip: self.boat_no_equip,
            boat_in2_rate: self.boat_in2_rate,
        }
    }
}

/// Race data container with lazy loading
pub struct RaceData {
    df: LazyFrame,
}

impl RaceData {
    /// Load race data from CSV file
    pub fn load<P: AsRef<Path>>(csv_path: P) -> Result<Self, PolarsError> {
        let df = LazyCsvReader::new(csv_path).finish()?;
        Ok(Self { df })
    }

    /// Get entries for a specific race
    pub fn get_race(
        &self,
        date: u32,
        stadium_code: u8,
        race_no: u8,
    ) -> Result<Vec<ProgramEntry>, PolarsError> {
        let filtered = self
            .df
            .clone()
            .filter(
                col("date")
                    .eq(lit(date as i64))
                    .and(col("stadium_code").eq(lit(stadium_code as i64)))
                    .and(col("race_no").eq(lit(race_no as i64))),
            )
            .collect()?;

        Self::dataframe_to_entries(&filtered)
    }

    /// Get all races for a specific date
    pub fn get_races_by_date(
        &self,
        date: u32,
    ) -> Result<HashMap<(u8, u8), Vec<ProgramEntry>>, PolarsError> {
        let filtered = self
            .df
            .clone()
            .filter(col("date").eq(lit(date as i64)))
            .collect()?;

        let entries = Self::dataframe_to_entries(&filtered)?;

        // Group by (stadium_code, race_no)
        let mut races: HashMap<(u8, u8), Vec<ProgramEntry>> = HashMap::new();
        for entry in entries {
            races
                .entry((entry.stadium_code, entry.race_no))
                .or_default()
                .push(entry);
        }

        Ok(races)
    }

    /// List available dates
    pub fn list_dates(&self) -> Result<Vec<u32>, PolarsError> {
        let dates = self
            .df
            .clone()
            .select([col("date")])
            .unique(None, UniqueKeepStrategy::First)
            .sort(["date"], Default::default())
            .collect()?;

        let date_col = dates.column("date")?.i64()?;
        Ok(date_col.into_no_null_iter().map(|d| d as u32).collect())
    }

    /// List races for a date (returns (stadium_code, race_no, num_entries) tuples)
    pub fn list_races(&self, date: u32) -> Result<Vec<(u8, u8, usize)>, PolarsError> {
        let filtered = self
            .df
            .clone()
            .filter(col("date").eq(lit(date as i64)))
            .group_by([col("stadium_code"), col("race_no")])
            .agg([col("boat_no").count().alias("count")])
            .sort(["stadium_code", "race_no"], SortMultipleOptions::default())
            .collect()?;

        let stadium_col = filtered.column("stadium_code")?.i64()?;
        let race_col = filtered.column("race_no")?.i64()?;
        let count_col = filtered.column("count")?.u32()?;

        let mut result = Vec::new();
        for i in 0..filtered.height() {
            if let (Some(s), Some(r), Some(c)) =
                (stadium_col.get(i), race_col.get(i), count_col.get(i))
            {
                result.push((s as u8, r as u8, c as usize));
            }
        }

        Ok(result)
    }

    /// Convert DataFrame to ProgramEntry vector
    fn dataframe_to_entries(df: &DataFrame) -> Result<Vec<ProgramEntry>, PolarsError> {
        let mut entries = Vec::with_capacity(df.height());

        // Use i64 for all integer columns (polars default inference)
        let date_col = df.column("date")?.i64()?;
        let stadium_col = df.column("stadium_code")?.i64()?;
        let race_col = df.column("race_no")?.i64()?;
        let boat_col = df.column("boat_no")?.i64()?;
        let racer_id_col = df.column("racer_id")?.i64()?;
        let racer_name_col = df.column("racer_name")?.str()?;
        let age_col = df.column("age")?.i64()?;
        let branch_col = df.column("branch")?.str()?;
        let weight_col = df.column("weight")?.i64()?;
        let class_col = df.column("racer_class")?.str()?;
        let nat_win_col = df.column("national_win_rate")?.f64()?;
        let nat_in2_col = df.column("national_in2_rate")?.f64()?;
        let local_win_col = df.column("local_win_rate")?.f64()?;
        let local_in2_col = df.column("local_in2_rate")?.f64()?;
        let motor_no_col = df.column("motor_no")?.i64()?;
        let motor_rate_col = df.column("motor_in2_rate")?.f64()?;
        let boat_equip_col = df.column("boat_no_equip")?.i64()?;
        let boat_rate_col = df.column("boat_in2_rate")?.f64()?;

        for i in 0..df.height() {
            entries.push(ProgramEntry {
                date: date_col.get(i).unwrap_or(0) as u32,
                stadium_code: stadium_col.get(i).unwrap_or(0) as u8,
                race_no: race_col.get(i).unwrap_or(0) as u8,
                boat_no: boat_col.get(i).unwrap_or(0) as u8,
                racer_id: racer_id_col.get(i).unwrap_or(0) as u32,
                racer_name: racer_name_col.get(i).unwrap_or("").to_string(),
                age: age_col.get(i).unwrap_or(0) as u8,
                branch: branch_col.get(i).unwrap_or("").to_string(),
                weight: weight_col.get(i).unwrap_or(0) as u8,
                racer_class: class_col.get(i).unwrap_or("").to_string(),
                national_win_rate: nat_win_col.get(i).unwrap_or(0.0),
                national_in2_rate: nat_in2_col.get(i).unwrap_or(0.0),
                local_win_rate: local_win_col.get(i).unwrap_or(0.0),
                local_in2_rate: local_in2_col.get(i).unwrap_or(0.0),
                motor_no: motor_no_col.get(i).unwrap_or(0) as u32,
                motor_in2_rate: motor_rate_col.get(i).unwrap_or(0.0),
                boat_no_equip: boat_equip_col.get(i).unwrap_or(0) as u32,
                boat_in2_rate: boat_rate_col.get(i).unwrap_or(0.0),
            });
        }

        Ok(entries)
    }
}

/// Race key for indexing: (date, stadium_code, race_no)
pub type RaceKey = (u32, u8, u8);

/// Indexed race data with O(1) lookups
///
/// Pre-loads all data into memory and indexes by (date, stadium_code, race_no)
/// for fast backtest iteration.
pub struct IndexedRaceData {
    /// All entries indexed by race key
    races: HashMap<RaceKey, Vec<ProgramEntry>>,
    /// All dates in sorted order
    dates: Vec<u32>,
}

impl IndexedRaceData {
    /// Load and index all race data from CSV
    pub fn load<P: AsRef<Path>>(csv_path: P) -> Result<Self, PolarsError> {
        let df = CsvReadOptions::default()
            .try_into_reader_with_file_path(Some(csv_path.as_ref().to_path_buf()))?
            .finish()?;

        let entries = RaceData::dataframe_to_entries(&df)?;

        // Build index
        let mut races: HashMap<RaceKey, Vec<ProgramEntry>> = HashMap::new();
        let mut date_set = std::collections::BTreeSet::new();

        for entry in entries {
            date_set.insert(entry.date);
            races
                .entry((entry.date, entry.stadium_code, entry.race_no))
                .or_default()
                .push(entry);
        }

        let dates: Vec<u32> = date_set.into_iter().collect();

        Ok(Self { races, dates })
    }

    /// Get entries for a specific race - O(1)
    pub fn get_race(&self, date: u32, stadium_code: u8, race_no: u8) -> Option<&Vec<ProgramEntry>> {
        self.races.get(&(date, stadium_code, race_no))
    }

    /// Get all dates in sorted order
    pub fn dates(&self) -> &[u32] {
        &self.dates
    }

    /// Get all races for a date - O(n) where n = races on that date
    pub fn get_races_by_date(&self, date: u32) -> HashMap<(u8, u8), &Vec<ProgramEntry>> {
        self.races
            .iter()
            .filter(|((d, _, _), _)| *d == date)
            .map(|((_, s, r), entries)| ((*s, *r), entries))
            .collect()
    }

    /// Iterate over all races
    pub fn iter(&self) -> impl Iterator<Item = (&RaceKey, &Vec<ProgramEntry>)> {
        self.races.iter()
    }

    /// Total number of races
    pub fn len(&self) -> usize {
        self.races.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.races.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_program_entry_to_racer_entry() {
        let entry = ProgramEntry {
            date: 20240115,
            stadium_code: 23,
            race_no: 1,
            boat_no: 1,
            racer_id: 5017,
            racer_name: "Test Racer".to_string(),
            age: 25,
            branch: "Tokyo".to_string(),
            weight: 52,
            racer_class: "A1".to_string(),
            national_win_rate: 6.95,
            national_in2_rate: 52.34,
            local_win_rate: 7.23,
            local_in2_rate: 61.54,
            motor_no: 36,
            motor_in2_rate: 42.61,
            boat_no_equip: 76,
            boat_in2_rate: 43.21,
        };

        let racer = entry.to_racer_entry();

        assert_eq!(racer.boat_no, 1);
        assert_eq!(racer.racer_id, 5017);
        assert_eq!(racer.racer_class, "A1");
        assert!((racer.national_win_rate - 6.95).abs() < 0.01);
    }
}
