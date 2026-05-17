import cv2
import yaml
import os


class ConfigTuner:
    def __init__(self, window_name="controls", config_path="line_config.yaml"):
        self.window_name = window_name
        self.config_path = config_path

        # Дефолтные значения (на случай, если файла нет)
        self.config = {
            "LB": 0, "LG": 0, "LR": 215,
            "HB": 148, "HG": 234, "HR": 255
        }

        self._load_config()
        self._init_trackbars()

    def _load_config(self):
        """Загрузка конфига из YAML, если файл существует."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    loaded_conf = yaml.safe_load(f)
                    if loaded_conf:
                        self.config.update(loaded_conf)
                print(f"[ConfigTuner] Конфигурация загружена из {self.config_path}")
            except Exception as e:
                print(f"[ConfigTuner] Ошибка чтения конфига: {e}")

    def save_config(self):
        """Сохранение текущих значений в YAML."""
        try:
            with open(self.config_path, 'w') as f:
                yaml.dump(self.config, f)
            print(f"[ConfigTuner] Конфигурация сохранена в {self.config_path}")
        except Exception as e:
            print(f"[ConfigTuner] Ошибка сохранения конфига: {e}")

    def _nothing(self, _):
        pass

    def _init_trackbars(self):
        """Создание окна и трекбаров."""
        cv2.namedWindow(self.window_name)
        for key in ["LB", "LG", "LR", "HB", "HG", "HR"]:
            cv2.createTrackbar(key, self.window_name, self.config[key], 255, self._nothing)

    def get_values(self):
        """Считывание текущих значений с ползунков и обновление внутреннего состояния."""
        values = {}
        for key in self.config.keys():
            val = cv2.getTrackbarPos(key, self.window_name)
            values[key] = val
            self.config[key] = val
        return values
