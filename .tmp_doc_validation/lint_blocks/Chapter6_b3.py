from pydantic import BaseModel
from artana.kernel import ApprovalRequiredError

class SendInvoiceArgs(BaseModel):
    billing_period: str
    amount_usd: float

try:
    await kernel.step_tool(
        run_id="billing_run",
        tenant=tenant,
        tool_name="send_invoice",
        arguments=SendInvoiceArgs(billing_period="2026-02", amount_usd=120.0),
        step_key="invoice_send",
    )
except ApprovalRequiredError as exc:
    await kernel.approve_tool_call(
        run_id="billing_run",
        tenant=tenant,
        approval_key=exc.approval_key,
        mode="human",
        reason="Finance manager approved",
    )
    await kernel.step_tool(
        run_id="billing_run",
        tenant=tenant,
        tool_name="send_invoice",
        arguments=SendInvoiceArgs(billing_period="2026-02", amount_usd=120.0),
        step_key="invoice_send",
    )