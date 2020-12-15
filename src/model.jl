export SegmentationModel

"""
segmentater of image
"""
struct SegmentationModel
    encoder::Chain
    decoder::Chain
end

function SegmentationModel()
    k₁ = (32, 32); s₁ = (16, 16);
    encoder = Chain(
        Conv(k₁, 3 => 3, relu, stride=s₁)
    )
    decoder = Chain(
        ConvTranspose(k₁, 3 => 3, relu, stride=s₁)
    )
    return SegmentationModel(encoder, decoder)
end

function (m::SegmentationModel)(x::Array)
    z = m.encoder(x)
    y = m.decoder(z)
    return y
end

function lossfunction(model)
    loss(x, y) = Flux.Losses.mse(model(x), y)
    return loss
end