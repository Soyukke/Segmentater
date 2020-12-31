export SegmentationModel

"""
segmentater of image
"""
struct SegmentationModel
    encoder::Chain
    decoder::Chain
end

function SegmentationModel(;n₁=32, n₂=16, n₃=8, m₁=4, m₂=2, m₃=1)
    k₁ = (n₁, n₁); s₁ = (m₁, m₁);
    k₂ = (n₂, n₂); s₂ = (m₂, m₂);
    k₃ = (n₃, n₃); s₃ = (m₃, m₃);
    encoder = Chain(
        Conv(k₁, 3 => 3, relu, stride=s₁),
        Conv(k₂, 3 => 3, relu, stride=s₂),
        Conv(k₃, 3 => 3, relu, stride=s₃)
    )
    decoder = Chain(
        ConvTranspose(k₁, 3 => 3, relu, stride=s₁),
        ConvTranspose(k₂, 3 => 3, relu, stride=s₂),
        ConvTranspose(k₃, 3 => 3, relu, stride=s₃)
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