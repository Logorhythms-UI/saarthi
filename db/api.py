from schema import db, ImageHistory
from tabulate import tabulate
import logging


class DB_API:
    def initialize(self) -> None:
        # Initialize the DB and create necessary tables
        db.connect()
        db.create_tables(
            [ImageHistory]
        )  # Safe creates the table (does not re-create if table already there)
        logging.info("Db Connection Esthablished, and Tables Ready")

    def finalize(self) -> None:
        db.close()

    def print_table(self) -> None:
        ans = ImageHistory.select()
        images = []
        headers = ["Image Name", "Image Path", "Objects Identified", "Timestamp"]

        for a in ans:
            images.append(
                [
                    a.img_name,
                    f"$PROJECT_HOME/storage/{a.img_name}.png",
                    (
                        "No Objects Identified"
                        if a.objects_identified is None
                        else a.objects_identified.split(":")
                    ),
                    a.timestamp,
                ]
            )

        print(tabulate(images, headers=headers, tablefmt="grid"))

    def add_image(self, objects_identified: list[str]) -> None:
        assert type(objects_identified) is list[str], (
            f"add_image expected a list of string objects identified, got {type(objects_identified)}."
        )
        record = ImageHistory.create(objects_identified=objects_identified)
        if record:
            logging.info(f"New Image with objects {objects_identified} was added.")
        else:
            logging.warning(
                f"Could not add new image with objects identified {objects_identified}"
            )


DB = DB_API()
DB.initialize()
DB.print_table()
DB.finalize()
