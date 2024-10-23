#3D Slicer requires absolute paths to the files to be loaded
#Before running the transformation, replace the root path with the path to the project folder

MAIN_PATH = None

assert MAIN_PATH is not None, "Replace the root path with the path to the project folder"

transform = slicer.util.loadTransform(MAIN_PATH + "/LinearTransform.tfm")


for i in range(1,41):
    print(f"Transforming {i:02d}")
    segmentationNode = slicer.util.loadSegmentation(f"{MAIN_PATH}/data/segthor_train/train/Patient_{i:02d}/GT.nii.gz")
    newSegment = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
    volumeNode = slicer.util.loadVolume(f"{MAIN_PATH}/data/segthor_train/train/Patient_{i:02d}/Patient_{i:02d}.nii.gz")

    sourceSegmentName = "Segment_2"
    segmentation = segmentationNode.GetSegmentation()
    sourceSegmentId = segmentation.GetSegmentIdBySegmentName(sourceSegmentName)
    newSegment.GetSegmentation().CopySegmentFromSegmentation(segmentation, sourceSegmentId)
    segmentation.RemoveSegment(sourceSegmentName)

    #Apply transform matrix and harden
    newSegment.SetAndObserveTransformNodeID(transform.GetID())
    newSegment.HardenTransform()

    segmentation.CopySegmentFromSegmentation(newSegment.GetSegmentation(), sourceSegmentId)

    newSegment.RemoveSegment(sourceSegmentName)
    for j in range(1, segmentation.GetNumberOfSegments()+1):
        newSegment.GetSegmentation().CopySegmentFromSegmentation(segmentation, f"Segment_{j}")

    labelmapVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
    slicer.modules.segmentations.logic().ExportVisibleSegmentsToLabelmapNode(newSegment, labelmapVolumeNode, volumeNode)

    #Export segmentation as nifti
    slicer.util.exportNode(labelmapVolumeNode, f"{MAIN_PATH}/data/train/Patient_{i:02d}/GT.nii.gz")

    #Clean up
    slicer.mrmlScene.RemoveNode(labelmapVolumeNode)
    slicer.mrmlScene.RemoveNode(newSegment)
    slicer.mrmlScene.RemoveNode(segmentationNode)
    slicer.mrmlScene.RemoveNode(volumeNode)
