
class PoseEvaluator:

    def __init__(self):

        # Example target ranges.
        # Replace these with ranges for your selected yoga pose.

        self.target_ranges = {
            "left_elbow": (80, 110),
            "right_elbow": (80, 110),

            "left_shoulder": (70, 120),
            "right_shoulder": (70, 120),

            "left_hip": (70, 120),
            "right_hip": (70, 120),

            "left_knee": (80, 110),
            "right_knee": (80, 110),

            "left_ankle": (70, 120),
            "right_ankle": (70, 120)
        }

    # Evaluate every calculated joint angle
    def evaluate_pose(self, angles):

        results = {}

        for joint_name, angle in angles.items():

            if angle is None:

                results[joint_name] = "Not visible"
                continue

            if joint_name not in self.target_ranges:

                results[joint_name] = "No target configured"
                continue

            minimum, maximum = self.target_ranges[joint_name]

            if minimum <= angle <= maximum:

                results[joint_name] = "Within target range"

            else:

                results[joint_name] = "Outside target range"

        return results

    # Generate feedback messages
    def get_feedback(self, evaluation_results):

        feedback = []

        for joint_name, status in evaluation_results.items():

            if status == "Not visible":

                feedback.append(
                    f"{joint_name}: Joint not clearly visible."
                )

            elif status == "Outside target range":

                feedback.append(
                    f"{joint_name}: Adjust your alignment."
                )

        if not feedback:
            feedback.append("All visible joints are within their configured ranges.")

        return feedback
