export img2matrix, getbatchdata, indicesloader

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
`dict` is `index` => `filename`.
"""
function getdataset(dict, indices)
    fns = map(indices) do i
        dict[i]
    end
    matrixs = Array{Float32, 3}[]
    for i in indices
        img = Images.load(fns[i])
        mat = img2matrix(img)
        push!(matrixs, mat)
    end
    matrix = cat(matrixs..., dims=4)
    return matrix
end


"""
indices DataLoader.
"""
function indicesloader(ndata::Integer; batchsize, shuffle=true)
    return DataLoader(1:ndata, batchsize=batchsize, shuffle=shuffle)
end

"""
get train input data `x` and output `y` data. `indices`
"""
function getbatchdata(x_dict, y_dict, indices)
    x = getdataset(x_dict, indices)
    y = getdataset(y_dict, indices)
    return x, y
end