import annotator.canny
import annotator.hed
import annotator.lineart
import annotator.lineart_anime
import annotator.midas
import annotator.openpose
import annotator.mlsd

annotators = {
    "Canny": annotator.canny.CannyDetector,
    "Softedge": annotator.hed.HEDdetector,
    "Lineart": annotator.lineart.LineartDetector,
    "Anime": annotator.lineart_anime.LineartAnimeDetector,
    "Depth": annotator.midas.MidasDetector,
    "Pose": annotator.openpose.OpenposeDetector,
    "M-LSD": annotator.mlsd.MLSDdetector,
}