class SkincareEngineException(Exception):
    """Classe base per le eccezioni del Clinical Skincare Engine."""
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


class FaceCountException(SkincareEngineException):
    """Sollevata quando il numero di volti rilevati non è esattamente 1."""
    def __init__(self, code: str, message: str = "Face detection quality gate failed"):
        # code deve essere "NO_FACE_DETECTED" o "MULTIPLE_FACES_DETECTED"
        super().__init__(code, message)


class PoseAngleException(SkincareEngineException):
    """Sollevata quando l'inclinazione della testa supera il limite consentito."""
    def __init__(self, message: str = "Head tilt exceeds allowed limits"):
        super().__init__("EXCESSIVE_HEAD_TILT", message)


class InvalidImageException(SkincareEngineException):
    """Sollevata quando l'immagine caricata è corrotta o non valida."""
    def __init__(self, message: str = "Invalid or corrupt image payload"):
        super().__init__("INVALID_IMAGE_PAYLOAD", message)


class DatabaseBusyException(SkincareEngineException):
    """Sollevata quando il database è temporaneamente bloccato."""
    def __init__(self, message: str = "Database is temporarily busy"):
        super().__init__("DATABASE_BUSY", message)
