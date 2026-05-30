//! Input validation utilities

use anyhow::{bail, Result};

/// Validate date in YYYYMMDD format
pub fn validate_date(date: u32) -> Result<()> {
    let s = date.to_string();
    if s.len() != 8 {
        bail!("Date must be 8 digits (YYYYMMDD), got: {}", date);
    }

    let year = date / 10000;
    let month = (date / 100) % 100;
    let day = date % 100;

    // Year bounds (data available 2020-present)
    if year < 2020 || year > 2030 {
        bail!("Year {} out of valid range [2020-2030]", year);
    }

    if !(1..=12).contains(&month) {
        bail!("Month {} invalid (must be 1-12)", month);
    }

    let days_in_month = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if (year % 4 == 0 && year % 100 != 0) || year % 400 == 0 {
                29
            } else {
                28
            }
        }
        _ => unreachable!(),
    };

    if !(1..=days_in_month).contains(&day) {
        bail!(
            "Day {} invalid for month {} (max {})",
            day,
            month,
            days_in_month
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_date_valid() {
        assert!(validate_date(20240115).is_ok()); // Normal date
        assert!(validate_date(20240229).is_ok()); // Leap year Feb 29
        assert!(validate_date(20230228).is_ok()); // Non-leap year Feb 28
        assert!(validate_date(20201231).is_ok()); // End of year
        assert!(validate_date(20200101).is_ok()); // Start of valid range
    }

    #[test]
    fn test_validate_date_invalid_format() {
        assert!(validate_date(2024011).is_err()); // 7 digits
        assert!(validate_date(202401150).is_err()); // 9 digits
        assert!(validate_date(0).is_err()); // Zero
    }

    #[test]
    fn test_validate_date_invalid_year() {
        assert!(validate_date(20190115).is_err()); // Too old
        assert!(validate_date(20310115).is_err()); // Too far in future
    }

    #[test]
    fn test_validate_date_invalid_month() {
        assert!(validate_date(20240015).is_err()); // Month 0
        assert!(validate_date(20241315).is_err()); // Month 13
    }

    #[test]
    fn test_validate_date_invalid_day() {
        assert!(validate_date(20240100).is_err()); // Day 0
        assert!(validate_date(20240132).is_err()); // Jan 32
        assert!(validate_date(20240230).is_err()); // Feb 30
        assert!(validate_date(20230229).is_err()); // Feb 29 non-leap year
        assert!(validate_date(20240431).is_err()); // Apr 31
    }
}
