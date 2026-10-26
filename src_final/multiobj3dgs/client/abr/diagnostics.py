"""Optional ABR status hooks in the production native playback panel."""
import time

from multiobj3dgs.client.viewer.window import NativeWindow


class ABRWindow(NativeWindow):
    """Inherit every native control, draw call and memory diagnostic unchanged."""

    def __init__(self, *args, diagnostics=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.abr_diagnostics = diagnostics
        self._diagnostic_at = 0.
        self._diagnostic = None

    def _ui_diagnostics(self):
        now = time.monotonic()
        if self._diagnostic is None or now >= self._diagnostic_at:
            self._diagnostic = self.runtime.diagnostics()
            self._diagnostic_at = now + .25
        return self._diagnostic

    def _draw_extra_ui(self, diagnostic):
        im = self.imgui
        _, self.abr_diagnostics = im.checkbox('ABR diagnostics', self.abr_diagnostics)
        if not self.abr_diagnostics:
            return
        abr = diagnostic.get('abr', {})
        decision = abr.get('last_decision') or {}
        network = diagnostic.get('network', {})
        im.text(f'Predicted: {decision.get("predicted_bandwidth_bps", 0)/1e6:.2f} Mb/s | budget: {decision.get("budget_bps", 0)/1e6:.2f}')
        im.text(f'Observed: {(network.get("observed_bps") or 0)/1e6:.2f} Mb/s | transfers active: {network.get("active_requests", 0)} | received: {network.get("total_bytes", 0)} B')
        im.text(f'Plan: {decision.get("status", "waiting")} | {decision.get("duration_s", 0)*1000:.3f} ms')
        im.text(f'GOP cost upper bound: {decision.get("estimated_rate_bps", 0)/1e6:.2f} Mb/s')
        for instance in self.runtime.catalog.instances:
            value = decision.get('selected', {}).get(instance.id, {})
            weight = abr.get('visibility_weights', {}).get(instance.id)
            selected = self.runtime.selector.selected[instance.id]
            im.text(f'{instance.id}: active={abr.get("active_representations", {}).get(instance.id)} selected={selected} | visibility={weight if weight is not None else "bootstrap"}')
            if value:
                im.text(f'  GOP {value["gop_index"]} PSNR={value["psnr"]:.3f} dB | missing={value["missing_bytes"]} B')
