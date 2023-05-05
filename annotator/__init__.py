import annotator.canny
import annotator.hed
import annotator.lineart
import annotator.lineart_anime
import annotator.midas
import annotator.openpose
import annotator.mlsd

annotators = {
    "canny": annotator.canny.CannyDetector,
    "softedge": annotator.hed.HEDdetector,
    "lineart": annotator.lineart.LineartDetector,
    "anime": annotator.lineart_anime.LineartAnimeDetector,
    "depth": annotator.midas.MidasDetector,
    "pose": annotator.openpose.OpenposeDetector,
    "mlsd": annotator.mlsd.MLSDdetector
}