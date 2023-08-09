import earth2mip.initial_conditions.era5 as initial_conditions
from earth2mip.initial_conditions import get_data_source
import datetime
import pytest
import pathlib
from earth2mip import schema, config
import xarray


@pytest.mark.parametrize("year", [2018, 1980, 2017])
def test__get_path(year):
    root = pathlib.Path(__file__).parent / "mock_data"
    root = root.as_posix()
    dt = datetime.datetime(year, 1, 2)
    path = initial_conditions._get_path(root, dt)
    assert pathlib.Path(path).name == dt.strftime("%Y.h5")
    assert pathlib.Path(path).exists()


def test__get_path_key_error():

    with pytest.raises(KeyError):
        initial_conditions._get_path(".", datetime.datetime(2040, 1, 2))


@pytest.mark.slow
def test_initial_conditions_get():
    time = datetime.datetime(2018, 1, 1)

    if not config.ERA5_HDF5_73:
        pytest.skip("No data location configured.")

    data = get_data_source(
        n_history=0,
        channel_set=schema.ChannelSet.var73,
        initial_condition_source=schema.InitialConditionSource.era5,
        grid=schema.Grid.grid_721x1440,
    )

    ic = data[time]
    assert isinstance(ic, xarray.DataArray)
