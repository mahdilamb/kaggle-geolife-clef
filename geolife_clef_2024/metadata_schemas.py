"""Schemas for the metadata."""

from collections.abc import Mapping
from types import MappingProxyType

import polars as pl

from geolife_clef_2024 import _type_aliases

P0_TRAIN: Mapping[
    _type_aliases.P0Feature | _type_aliases.TargetFeature, pl.PolarsDataType
] = MappingProxyType(
    {
        "publisher": pl.Enum(
            (
                "Slovenian Forestry Institute",
                "SLU Artdatabanken",
                "iNaturalist.org",
                "Botanical Society of Britain & Ireland",
                "Miljøstyrelsen / The Danish Environmental Protection Agency",
                "National Plant Monitoring Scheme",
                "Pl@ntNet",
                "Masaryk University, Department of Botany and Zoology",
                "Swiss National Biodiversity Data and Information Centres – "
                + "infospecies.ch",
                "The Norwegian Biodiversity Information Centre (NBIC)",
                "Observation.org",
                "Natuurpunt",
            )
        ),
        "year": pl.Int64,
        "month": pl.Int64,
        "day": pl.Int64,
        "lat": pl.Float64,
        "lon": pl.Float64,
        "geoUncertaintyInM": pl.Float64,
        "taxonRank": pl.Enum(("SPECIES", "SUBSPECIES")),
        "date": pl.Date,
        "dayOfYear": pl.Int64,
        "surveyId": pl.Int64,
        "speciesId": pl.Int64,
    }
)
PA_TEST: Mapping[
    _type_aliases.PAFeature | _type_aliases.TargetFeature, pl.PolarsDataType
] = MappingProxyType(
    {
        "lon": pl.Float64,
        "lat": pl.Float64,
        "year": pl.Int64,
        "geoUncertaintyInM": pl.Float64,
        "areaInM2": pl.Float64,
        "region": pl.Enum(
            (
                "ALPINE",
                "ATLANTIC",
                "BLACK SEA",
                "BOREAL",
                "CONTINENTAL",
                "MEDITERRANEAN",
                "PANNONIAN",
                "STEPPIC",
            )
        ),
        "country": pl.Enum(
            (
                "Portugal",
                "France",
                "Greece",
                "Serbia",
                "Slovakia",
                "Bulgaria",
                "Norway",
                "Slovenia",
                "Monaco",
                "Czech Republic",
                "Switzerland",
                "Bosnia and Herzegovina",
                "Croatia",
                "Netherlands",
                "Poland",
                "Andorra",
                "Germany",
                "Belgium",
                "Hungary",
                "Spain",
                "Italy",
                "Romania",
                "Luxembourg",
                "Ireland",
                "Montenegro",
                "Latvia",
                "The former Yugoslav Republic of Macedonia",
                "Austria",
                "Denmark",
            )
        ),
        "surveyId": pl.Int64,
    }
)

PA_TRAIN: Mapping[
    _type_aliases.PAFeature | _type_aliases.TargetFeature,
    pl.PolarsDataType,
] = MappingProxyType(
    {
        "lon": pl.Float64,
        "lat": pl.Float64,
        "year": pl.Int64,
        "geoUncertaintyInM": pl.Float64,
        "areaInM2": pl.Float64,
        "region": pl.String,
        "country": pl.String,
        "surveyId": pl.Int64,
        "speciesId": pl.Int64,
    }
)
