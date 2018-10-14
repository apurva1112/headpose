from tracker import Tracker

tracker = Tracker(
    detector=("./deploy.prototxt.txt",
              "./res10_300x300_ssd_iter_140000.caffemodel"),
    confidence=0.4,
    tracker='kcf',
    refresh_interval=5,
    video=None,
    width=400,
    window=None
  )

tracker.start()
