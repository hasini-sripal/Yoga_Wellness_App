
import json
from pathlib import Path


class PoseEvaluator:

    def __init__(self):

        ROOT = Path(__file__).resolve().parent.parent

        reference_path = ROOT / "models" / "reference_angles.json"

        with open(reference_path, "r") as file:
            self.reference_data = json.load(file)

        self.references = self.reference_data["ref"]
        self.tolerances = self.reference_data["tol"]

    def evaluate_pose(self, pose_name, angles):

        results = {}

        # Check whether the predicted pose has reference data
        if pose_name not in self.references:
            return {
                "error": "No reference angles available for this pose."
            }

        reference = self.references[pose_name]
        tolerance = self.tolerances[pose_name]

        for joint_name, current_angle in angles.items():

            if current_angle is None:
                results[joint_name] = "Not visible"
                continue

            column = "angle_" + joint_name

            reference_angle = reference.get(column)

            allowed_deviation = tolerance.get(column)

            if reference_angle is None:
                results[joint_name] = "No reference available"
                continue

            # Handle missing or invalid standard deviation
            if allowed_deviation is None:
                allowed_deviation = 15

            if allowed_deviation != allowed_deviation:
                allowed_deviation = 15

            # Avoid an unrealistically narrow tolerance
            allowed_deviation = max(
                10,
                2 * allowed_deviation
            )

            difference = abs(
                current_angle - reference_angle
            )

            if difference <= allowed_deviation:
                results[joint_name] = "Good alignment"
            else:
                results[joint_name] = "Needs adjustment"

        return results

    def get_feedback(self, evaluation_results):

        feedback = []

        for joint_name, status in evaluation_results.items():

            if status == "Needs adjustment":
                feedback.append(
                    f"{joint_name}: Adjust your alignment."
                )

            elif status == "Not visible":
                feedback.append(
                    f"{joint_name}: Joint is not clearly visible."
                )

        if not feedback:
            feedback.append("No alignment issues detected.")

        return feedback
