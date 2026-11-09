"""Bundle only the privately loaded Material Design font used by Pitward."""

from PyInstaller.utils.hooks import collect_data_files

datas = collect_data_files("qtawesome", includes=["fonts/materialdesignicons6-*"])
