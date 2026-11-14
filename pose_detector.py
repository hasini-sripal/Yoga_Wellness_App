
class PoseEvaluator:

    def __init__(self, pose_name, angle_rules):
        self.pose_name = pose_name
        self.angle_rules = angle_rules
        self.feedback = []

    def evaluate_pose(self, angles):
        self.feedback = []

        for angle_name, rule in self.angle_rules.items():

            # Check whether the angle was detected
            if angle_name not in angles or angles[angle_name] is None:
                self.feedback.append(
                    f"Unable to detect {angle_name}."
                )
                continue

            current_angle = angles[angle_name]

            # Get the acceptable angle range
            min_angle = rule["min"]
            max_angle = rule["max"]

            # Compare the detected angle with the required range
            if min_angle <= current_angle <= max_angle:
                self.feedback.append(
                    f"{angle_name}: Good!"
                )
            elif current_angle < min_angle:
                self.feedback.append(
                    f"{angle_name}: Increase the angle slightly."
                )
            else:
                self.feedback.append(
                    f"{angle_name}: Decrease the angle slightly."
                )

        return self.feedback

    def get_feedback(self):
        return self.feedback

