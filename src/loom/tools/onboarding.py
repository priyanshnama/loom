"""User onboarding tools — replace stubs with real workflow triggers."""

from __future__ import annotations


async def onboard_user(email: str, name: str, plan: str = "free") -> str:
    """Initiate the onboarding workflow for a new user.

    Args:
        email: The user's email address.
        name: The user's full name.
        plan: Subscription plan — "free", "pro", or "enterprise".

    Returns:
        Confirmation that the onboarding workflow was triggered.
    """
    # TODO: integrate with your onboarding system (e.g. trigger a workflow,
    # send a welcome email, create a CRM record, etc.)
    return (
        f"[STUB] Onboarding triggered for {name} <{email}> on the '{plan}' plan. "
        "Replace this stub with a real workflow call."
    )
