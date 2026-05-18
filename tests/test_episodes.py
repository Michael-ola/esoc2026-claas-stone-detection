import pandas as pd
import pytest

from claas_stone_detection.episodes import extract_header_on_episodes


def make_episode_dataframe(header_on: list[bool]) -> pd.DataFrame:
    return pd.DataFrame(
        {"HeaderOn": header_on},
        index=pd.Index([float(i) for i in range(len(header_on))], name="time_s"),
    )


def test_extract_header_on_episodes_finds_single_episode() -> None:
    df = make_episode_dataframe([False, True, True, True, False])

    episodes = extract_header_on_episodes(df, grace_s=2.0)

    assert len(episodes) == 1
    assert episodes[0].start_time == 1.0
    assert episodes[0].end_time == 4.0
    assert episodes[0].extended_end_time == 4.0


def test_extract_header_on_episodes_finds_multiple_episodes() -> None:
    df = make_episode_dataframe([False, True, True, False, False, True, True, False])

    episodes = extract_header_on_episodes(df, grace_s=1.0)

    assert len(episodes) == 2

    assert episodes[0].start_time == 1.0
    assert episodes[0].end_time == 3.0
    assert episodes[0].extended_end_time == 4.0

    assert episodes[1].start_time == 5.0
    assert episodes[1].end_time == 7.0
    assert episodes[1].extended_end_time == 7.0


def test_extract_header_on_episodes_extends_end_time_with_grace_period() -> None:
    df = make_episode_dataframe([True, True, True, False, False, False, False])

    episodes = extract_header_on_episodes(df, grace_s=2.0)

    assert len(episodes) == 1
    assert episodes[0].start_time == 0.0
    assert episodes[0].end_time == 3.0
    assert episodes[0].extended_end_time == 5.0


def test_extract_header_on_episodes_does_not_extend_beyond_max_time() -> None:
    df = make_episode_dataframe([False, False, True, True, False])

    episodes = extract_header_on_episodes(df, grace_s=10.0)

    assert len(episodes) == 1
    assert episodes[0].extended_end_time == 4.0


def test_extract_header_on_episodes_filters_short_episodes() -> None:
    df = make_episode_dataframe([False, True, False, False, True, True, False])

    episodes = extract_header_on_episodes(df, min_duration_s=1.5, grace_s=1.0)

    assert len(episodes) == 1
    assert episodes[0].start_time == 4.0
    assert episodes[0].end_time == 6.0


def test_extract_header_on_episodes_handles_episode_open_at_end() -> None:
    df = make_episode_dataframe([False, True, True, True])

    episodes = extract_header_on_episodes(df, grace_s=2.0)

    assert len(episodes) == 1
    assert episodes[0].start_time == 1.0
    assert episodes[0].end_time == 3.0
    assert episodes[0].extended_end_time == 3.0


def test_extract_header_on_episodes_returns_empty_list_for_empty_dataframe() -> None:
    df = pd.DataFrame(
        {"HeaderOn": []},
        index=pd.Index([], name="time_s"),
    )

    episodes = extract_header_on_episodes(df)

    assert episodes == []


def test_extract_header_on_episodes_rejects_missing_header_on_column() -> None:
    df = pd.DataFrame(
        {"OtherColumn": [True, False]},
        index=pd.Index([0.0, 1.0], name="time_s"),
    )

    with pytest.raises(ValueError, match="Missing header-on column"):
        extract_header_on_episodes(df)