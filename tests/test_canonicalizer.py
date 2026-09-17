"""
Test Aspect Canonicalizer.
"""

from src.ingestion.canonicalizer import AspectCanonicalizer


def test_hotel_canonicalizer():
    canon = AspectCanonicalizer(domain="hotel")
    assert canon.canonicalize("hygiene") == "Cleanliness"
    assert canon.canonicalize("bathroom") == "Cleanliness"
    assert canon.canonicalize("reception desk") == "Service"
    assert canon.canonicalize("subway station") == "Location"
    assert canon.canonicalize("wifi connection") == "Amenities"
    assert canon.canonicalize("overpriced cost") == "Value"


def test_electronics_canonicalizer():
    canon = AspectCanonicalizer(domain="electronics")
    assert canon.canonicalize("battery drain") == "Battery"
    assert canon.canonicalize("screen-on-time") == "Battery"
    assert canon.canonicalize("amoled panel") == "Display"
    assert canon.canonicalize("gaming heating") == "Performance"
