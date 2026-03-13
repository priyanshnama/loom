"""Flight booking tools — replace stubs with real booking API calls."""

from __future__ import annotations


async def book_flight(origin: str, destination: str, date: str) -> str:
    """Book a flight ticket for the user.

    Args:
        origin: Departure airport code or city name (e.g. "DEL", "Delhi").
        destination: Arrival airport code or city name (e.g. "BOM", "Mumbai").
        date: Travel date in YYYY-MM-DD format.

    Returns:
        Booking confirmation string with PNR and details.
    """
    # TODO: integrate with booking API (Amadeus, Skyscanner, etc.)
    return (
        f"[STUB] Flight booked: {origin} → {destination} on {date}. "
        "PNR: LM-000000. Please replace this stub with a real API call."
    )


async def cancel_flight(pnr: str) -> str:
    """Cancel an existing flight booking by PNR.

    Args:
        pnr: The booking reference / PNR to cancel.

    Returns:
        Cancellation confirmation string.
    """
    return f"[STUB] Flight {pnr} cancellation requested. Replace with real API."
