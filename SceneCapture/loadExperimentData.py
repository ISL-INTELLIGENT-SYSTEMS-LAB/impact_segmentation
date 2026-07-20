import json
import os

# Class to define colored output text styles for the terminal
class TxtColor:
    BOLD = '\033[1m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    ORANGE = '\033[93m'
    MAGENTA = '\033[95m'
    BLUE = '\033[94m'
    END = '\033[0m'  # Resets formatting

# Function to list and display details about experiment folders
def ListExperiments(PATH):
    experiments = []

    # Loop through each item in the specified directory
    for folder in os.listdir(PATH):
        full_path = os.path.join(PATH, folder)

        # Only process directories (ignore files)
        if os.path.isdir(full_path):
            experiments.append(folder)

            # Print the folder name in green and bold
            print(TxtColor.BOLD + TxtColor.GREEN + f"\n{folder}" + TxtColor.END)

            # Try to open and read the data.json file in this folder
            try:
                with open(os.path.join(full_path, "data.json"), "r") as f:
                    data = json.load(f)

                    try:
                        # Print each object in the mask legend and its grayscale value
                        for key in data["mask_legend"]:
                            print(TxtColor.MAGENTA + f"{key} RGB value: " + TxtColor.END, end="")
                            print(f"{data['mask_legend'][key]}")

                        # Print object information (positions)
                        objects = [obj for obj in data['Objects']]
                        for obj in objects:
                            print(TxtColor.ORANGE + f"{obj}" + TxtColor.END)
                            print(f"\txpos: {data['Objects'][obj]['xpos']}")
                            print(f"\typos: {data['Objects'][obj]['zpos']}")

                        # Print camera information (positions, rotation, and visible objects)
                        for cam in data['Cameras']:
                            print(TxtColor.RED + f"{cam}" + TxtColor.END)
                            print(f"\txpos: {data['Cameras'][cam]['xpos']}")
                            print(f"\typos: {data['Cameras'][cam]['zpos']}")
                            print(f"\tRotation: {data['Cameras'][cam]['rotation']}")
                            print(f"\tObjects: {data['Cameras'][cam]['objects']}")
                    except KeyError as e:
                        # Handle missing expected keys inside data.json
                        print(f"KeyError: {e} in folder {folder}")
                        continue

            except FileNotFoundError:
                # Skip folders that don't contain a data.json file
                print(f"FileNotFoundError: data.json not found in folder {folder}")
                continue

    # Return the total number of experiment folders processed
    return len(experiments)


# Entry point of the script
if __name__ == "__main__":
    PATH = "SceneCapture"  # Directory containing experiment folders
    experiments = ListExperiments(PATH)  # Run the listing function
    print(TxtColor.BLUE + TxtColor.BOLD + "\nTotal experiments found:" +
          TxtColor.RED + f" {experiments}" + TxtColor.END)
