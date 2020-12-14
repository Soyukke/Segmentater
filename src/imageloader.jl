export img2matrix

"""
RGB image (height, width, 3)
"""
function img2matrix(img)
    R = map(x -> x.r, img)
    G = map(x -> x.g, img)
    B = map(x -> x.b, img)
    return float(cat(R, G, B, dims=3))
end