package bifrost

import (
	"testing"

	schemas "github.com/maximhq/bifrost/core/schemas"
)

// B3: two isolated contexts derived from the same parent must NOT share a
// RequestID. Value fall-through would otherwise let both read an ID seeded on
// the parent, collapsing two calls onto one governance billing identity.
func TestPassthroughIsolatedCtxDistinctRequestID(t *testing.T) {
	parent, cancel := schemas.NewBifrostContextWithCancel(t.Context())
	defer cancel()
	parent.SetValue(schemas.BifrostContextKeyRequestID, "SEEDED-BY-EARLIER-CALL")

	a := isolatedPassthroughContext(parent)
	b := isolatedPassthroughContext(parent)

	idA, okA := a.Value(schemas.BifrostContextKeyRequestID).(string)
	idB, okB := b.Value(schemas.BifrostContextKeyRequestID).(string)
	t.Logf("parent=%q childA=%q childB=%q", "SEEDED-BY-EARLIER-CALL", idA, idB)

	if !okA || !okB {
		t.Fatalf("RequestID missing: okA=%v okB=%v", okA, okB)
	}
	if idA == "SEEDED-BY-EARLIER-CALL" || idB == "SEEDED-BY-EARLIER-CALL" {
		t.Fatalf("INHERITED the parent's RequestID: A=%q B=%q", idA, idB)
	}
	if idA == idB {
		t.Fatalf("COLLISION: both isolated contexts share RequestID %q", idA)
	}

	// Fall-through for other keys must still work.
	parent.SetValue(schemas.BifrostContextKeyVirtualKey, "vk-from-parent")
	if got, _ := a.Value(schemas.BifrostContextKeyVirtualKey).(string); got != "vk-from-parent" {
		t.Fatalf("value fall-through broken: VirtualKey=%q, want %q", got, "vk-from-parent")
	}
	t.Logf("distinct IDs and fall-through intact")
}
