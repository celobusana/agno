from pathlib import Path
from typing import Any, List, Optional, Union

from agno.tools import Toolkit
from agno.utils.log import log_debug, log_info, logger


class AirflowTools(Toolkit):
    def __init__(
        self, dags_dir: Optional[Union[Path, str]] = None, save_dag: bool = True, read_dag: bool = True, **kwargs
    ):
        """
        quick start to work with airflow : https://airflow.apache.org/docs/apache-airflow/stable/start.html
        """

        _dags_dir: Optional[Path] = None
        if dags_dir is not None:
            if isinstance(dags_dir, str):
                _dags_dir = Path.cwd().joinpath(dags_dir)
            else:
                _dags_dir = dags_dir
        self.dags_dir: Path = _dags_dir or Path.cwd()

        tools: List[Any] = []
        if save_dag:
            tools.append(self.save_dag_file)
        if read_dag:
            tools.append(self.read_dag_file)

        super().__init__(name="AirflowTools", tools=tools, **kwargs)

    def save_dag_file(self, contents: str, dag_file: str) -> str:
        """Saves python code for an Airflow DAG to a file called `dag_file` and returns the file path if successful.

        :param contents: The contents of the DAG.
        :param dag_file: The name of the file to save to.
        :return: The file path if successful, otherwise returns an error message.
        """
        try:
            file_path = self.dags_dir.joinpath(dag_file)
            log_debug(f"Saving contents to {file_path}")
            if not file_path.parent.exists():
                file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(contents)
            log_info(f"Saved: {file_path}")
            return str(str(file_path))
        except Exception as e:
            logger.error(f"Error saving to file: {e}")
            return f"Error saving to file: {e}"

    def read_dag_file(self, dag_file: str) -> str:
        """Reads an Airflow DAG file `dag_file` and returns the contents if successful.

        :param dag_file: The name of the file to read
        :return: The contents of the file if successful, otherwise returns an error message.
        """
        try:
            log_info(f"Reading file: {dag_file}")
            file_path = self.dags_dir.joinpath(dag_file)
            contents = file_path.read_text()
            return str(contents)
        except Exception as e:
            logger.error(f"Error reading file: {e}")
            return f"Error reading file: {e}"
