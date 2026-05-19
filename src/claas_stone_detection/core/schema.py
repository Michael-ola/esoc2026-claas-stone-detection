from dataclasses import dataclass


@dataclass(frozen=True)
class ChannelSchema:
    """Expected channel names in the CLAAS MF4 measurement files."""

    sensor: str = "Sensor1"
    vehicle_speed: str = "VehicleSpeed"
    cut_length: str = "CutLength"
    voltage: str = "VoltageSignal"
    status: str = "Status"
    header_on: str = "HeaderOn"

    @property
    def required_channels(self) -> tuple[str, ...]:
        return (
            self.sensor,
            self.vehicle_speed,
            self.cut_length,
            self.voltage,
            self.status,
        )


DEFAULT_SCHEMA = ChannelSchema()