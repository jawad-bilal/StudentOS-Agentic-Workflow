"""Expand recurring constraints into concrete datetime intervals."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from core.models import (
    AcademicInput,
    DateStr,
    StudySession,
    TimetableBlock,
    combine_local,
    parse_time_str,
)


def dates_in_horizon(academic_input: AcademicInput) -> list[date]:
    start = academic_input.reference_now.date()
    end = academic_input.horizon_end().date()
    days: list[date] = []
    current = start
    while current <= end:
        days.append(current)
        current += timedelta(days=1)
    return days


def weekday_for(d: date) -> int:
    return (d.weekday())  # Monday=0


def timetable_occurrences(
    block: TimetableBlock,
    academic_input: AcademicInput,
) -> list[tuple[date, datetime, datetime]]:
    tz = academic_input.timezone_info()
    out: list[tuple[date, datetime, datetime]] = []
    for d in dates_in_horizon(academic_input):
        if weekday_for(d) == int(block.weekday):
            start = combine_local(d.isoformat(), block.start, tz)
            end = combine_local(d.isoformat(), block.end, tz)
            out.append((d, start, end))
    return out


def session_interval(session: StudySession, tz: ZoneInfo) -> tuple[datetime, datetime]:
    start = combine_local(session.date, session.start, tz)
    end = combine_local(session.date, session.end, tz)
    return start, end


def intervals_overlap(a_start: datetime, a_end: datetime, b_start: datetime, b_end: datetime) -> bool:
    return a_start < b_end and b_start < a_end


def study_minutes_by_date(sessions: list[StudySession]) -> dict[DateStr, int]:
    totals: dict[DateStr, int] = {}
    for session in sessions:
        minutes = (
            parse_time_str(session.end).hour * 60
            + parse_time_str(session.end).minute
            - parse_time_str(session.start).hour * 60
            - parse_time_str(session.start).minute
        )
        totals[session.date] = totals.get(session.date, 0) + minutes
    return totals
