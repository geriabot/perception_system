# One open request of the perception system (in PerceptionStatus)

uint8 KIND_DETECTION=0
uint8 KIND_TARGET_FRAME=1
uint8 KIND_SKILL=2

string name
uint8 kind
# What it does, e.g. "person in red (also person); strongsort; /perception/attention"
string description
# Detection: the topics it publishes. Target frame: the TF frame. Skill: the skill's topics
string[] outputs
# Target frame: false while the followed detections are not seen. Detection and skill: true
bool ok
