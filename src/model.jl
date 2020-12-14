export SegmentationModel

"""
segmentater of image
"""
struct SegmentationModel
    encoder::Chain
    decoder::Chain
end

function SegmentationModel()
    encoder = Chain(
        Conv((128, 128), 3 => 2, relu, stride=(128, 128))
    )
    decoder = Chain(
        ConvTranspose((128, 128), 2 => 3, relu, stride=(128, 128))
    )
    return SegmentationModel(encoder, decoder)
end
