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


"""
get image files by indices of full dataset
`dict` is index => filename.
"""
function getdataset(dict, indices)
    fns = map(indices) do i
        dict[i]
    end
    matrixs = []
    for i in indices
        img = Images.load(fns[i])
        mat = img2matrix(img)
        push!(matrixs, mat)
    end
    matrix = cat(matrixs..., dims=4)
    return matrix
end
