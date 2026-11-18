# The perception of the user

The skills about the user are lifecycle nodes of `perception_user`, started with `user:=true`
and activated by `/perception_system` while a client requests them (`start_skill`) or from
activation (`autostart_skills`). While inactive they process no image.

| Skill | Node | Output | State |
|---|---|---|---|
| `speaking` | `/lip_activity_detector` | `/user_speaking` (`std_msgs/Bool`), `/lip_activity` (`std_msgs/Float32`) | Used for visual barge-in by the interaction system |
| `emotion` | `/emotion_detector` | `/user_emotion` (`std_msgs/String`, JSON) | Experimental, not validated |

`/perception_system/get_user_state` (`GetUserState`) returns the latest of both: `speaking`, and
the emotion with its confidence and source (`available` is false until an emotion has arrived).

## Lip activity

MediaPipe FaceLandmarker gives the gap of the lips of the face in each image, divided by the
height of the face (nose tip to chin). The node smooths it over `smoothing_frames` images and
computes its *roughness*: the variance of the consecutive changes over the last
`roughness_window` images. Speech moves the lips up and down several times per second and keeps
the roughness high; a single opening, a yawn or a smile moves them in one direction and the
roughness drops as soon as they stop. Speaking is reported after `onset_secs` of continuous high
roughness, and silence after `silence_secs` below it, so that a single movement does not count.
No face is a gap of 0. The face model is downloaded once to `~/.cache/perception_system` (or
`$PERCEPTION_MODELS_DIR`).

It looks at the largest face (one face is tracked): with several people in view, it is not the
user's lips unless the user is the one in front of the camera.

## Facial expression (experimental)

The largest face of each image (MediaPipe FaceDetector) is classified by a third-party image
classifier (`dima806/face_emotions_image_detection`, labels angry, disgust, fear, happy, neutral,
sad, surprise), one image at a time. It is not validated:

- each image is classified on its own, without smoothing over time;
- the classifier was trained on posed faces, and is unreliable with real faces, light, distance
  and movement;
- a facial expression is not a reliable sign of an emotion: use it, if at all, to adapt the
  manner of the robot, never to state what the person feels;
- in the EU, the AI Act restricts emotion recognition (prohibited in workplaces and education,
  with exceptions): check it, and obtain the ethics and data-protection approvals, before any use
  with people.
