
using BlackBoxOptim
using Images
using Interpolations, ImageCore, ImageTransformations
using CoordinateTransformations, Rotations, StaticArrays, OffsetArrays
using LinearAlgebra
using Optim
using Statistics
using FileIO
using FITSIO
using LsqFit
using Random, Distributions
using InterpolationKernels
using OptimPackNextGen
using EasyFITS
using OptimPackNextGen.Powell
using LocalFilters
using StatsBase
using ImageFiltering



"""
    Poly2D(
        order::Int  #order of the polynomial
        refpix::Tuple{Int,Int}   #reference pixel
        coef::Vector{Any} #polynomials coeficients  
        map::Tuple{AbstractVector{AbstractVector{Int64}},AbstractVector{AbstractVector{Int64}}} #Exponent Map
    )

Structure containing a 2D polynomial of arbitrary order.
coef is like : ax^2,bxy,cy^2,dx,ey,f (for Order = 2)
refpix is a reference pixel for mapping
"""
mutable struct Poly2D
    order::Int64  #order of the polynomial
    refpix::Tuple{Int64,Int64}   #reference pixel
    coef::Vector{Float64} #polynomials coeficients  
    map::Tuple{AbstractVector{Int64},AbstractVector{Int64}} #Exponent Map
end

"""
    Poly2D(order::Int,refpix::Tuple{Int,Int})

Generation of Poly2D object with automatic exponents
"""
Poly2D(order::Int64,refpix::Tuple{Int64,Int64},coef::Vector{Float64})=Poly2D(order::Int64,refpix::Tuple{Int64,Int64}, coef::Vector{Float64},ExpMap(order::Int64))



"""
    Poly2D(order::Int,refpix::Tuple{Int,Int})

Generation of Poly2D object with semi random coeficient
"""
Poly2D(order::Int64,refpix::Tuple{Int64,Int64})=Poly2D(order::Int64,refpix::Tuple{Int64,Int64},RandCoef(order))

"""
    RandCoef(order::Int)

Generate random coeficient to create a Poly2D object
"""
function RandCoef(order::Int64)
    coef = []
    for i in 0:1:order
        factor = (10.)^((order-i-1)*(-1))
        push!(coef,(rand(Float64,(order-i+1))).*factor)
    end
    return vcat(coef...)
end

"""
    ExpMap(coef::AbstractVector{AbstractVector{Float64}})

Set the maps of exponent order for the polynomial 

"""
function ExpMap(order::Int64)
    coef = []
    for i in 0:1:order
        factor = (10.)^((order-i)*(-10))
        push!(coef,(rand(Float64,(order-i+1))).*factor)
    end
    expx = convert(AbstractArray{AbstractArray{Int64}},copy(coef).*0)
    expy = copy(expx.*0)
    for i in 0:1:size(coef)[1]-1
        for j in 0:1:size(coef[i+1])[1]-1
            expx[i+1][j+1]=(size(coef[i+1])[1]-1-j)
            expy[i+1][j+1] = j
        end
    end
    map = (vcat(expx...),vcat(expy...))
    return map
end

"""
    (self::Poly2D)(x::Float,y::Float)

compute the value of the polyomial Poly2D at coordinates (x,y)

Example:

D = Poly2D(order,refpix,coef)
z = D(x,y)
"""
function (self::Poly2D)(x::Float64,y::Float64)
    return sum((x .^ self.map[1]) .* (y .^ self.map[2]) .* self.coef)
end

function (self::Poly2D)(x::Int64,y::Int64)
    return sum((x .^ self.map[1]) .* (y .^ self.map[2]) .* self.coef)
end
"""
    CoefConstraintIdentity!(poly::Poly2D,dim::Int64)

Change the coef value tu ensure identity projection in the dimension dim
"""
function CoefConstraintIdentity!(poly::Poly2D,dim::Int64)
    mask1 = findall(poly.map[dim] .== 1)
    mask2 = findall((poly.map[1] + poly.map[2]) .== 1)
    inter = intersect(mask1,mask2)
    poly.coef .= 0
    poly.coef[inter] .= 1
end

"""
    Residual(image1::Matrix{T},image2::Matrix{T}) where T
Return the residual image between image1 an image2
"""
function Residual(image1::Matrix{T},image2::Matrix{T}) where T
    if size(image1) != size(image2)
        image2 = imresize(image2,size(image1))       
    end
    return (image1 .- image2)
end


"""
    CoefConstraintCustom!(poly::Poly2D,dim::Int64)

Change the coef value tu ensure identity projection in the dimension dim
"""
function CoefConstraintCustom!(poly::Poly2D,dim::Int64)
    mask1 = findall(poly.map[dim] .== 1)
    mask2 = findall((poly.map[1] + poly.map[2]) .== 1)
    inter = intersect(mask1,mask2)

    poly.coef .= 0
    poly.coef[inter] .= 1
end

"""
    Contrast(image::Matrix{Float64},dim::Int)

Return the constrast of an image along the dimension Dim normalized by image lenght in dim dimension
"""
function Contrast(image::Matrix{Float64},dim::Int)
    return sum(image,dims=dim) #./ size(image)[dim]
end
function Contrast(image::Matrix{Float32},dim::Int)
    return sum(image,dims=dim) #./ size(image)[dim]
end


"""
    BadPixMap(fibre::Matrix{Float64},wave::Matrix{Float64})

Return a simple bad pixel map from fibre and wave frames. (bad pixels are flagged with 0 weight when outside a median filter) 
"""
function BadPixMap(fibre::Matrix{Float64},wave::Matrix{Float64})
    #bad pixel map for fibre
    for i in 1:size(fibre)[1]
        dam = mad(fibre[i,:])
        med = median(fibre[i,:])
        tresh = med + 5*dam
        for j in 1:size(fibre)[2]
            if fibre[i,j] > tresh
                badfibre[i,j] = 0.
            end
        end
    end
    #bad pixel map for wave
    for j in 1:size(wave)[2]
        dam = mad(wave[:,j])
        med = median(wave[:,j])
        tresh = med + 5*dam 
        for i in 1:size(wave)[1]
            if wave[i,j] > tresh
                badwave[i,j] = 0.
            end
        end
    end
    return badfibre,badwave
end

"""
Return a warped tranformed with the 2D Polynomial associated with Poly
"""
function ImageWarp(image::Matrix{Float64},Poly1::Poly2D,Poly2::Poly2D,axes::Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}})

    ϕ(x) = (Poly1(x[1],x[2]),Poly2(x[1],x[2]))
    Y = warp(image,ϕ,axes,fillvalue=0.,method=Interpolations.BSpline(Linear()))

    return Y
end

"""
Return a warped tranformed with the 2D Polynomial associated with Poly
"""
function ImageWarp2(image::Matrix{Float64},wgt::Matrix{Float64},Poly1::Poly2D,Poly2::Poly2D)

    ϕ(x) = (Poly1(x[1],x[2]),Poly2(x[1],x[2]))
    ker = CatmullRomSpline()    
    I = size(image)[1]
    J = size(image)[2]
    imwarped = zeros(I,J)
    range = 5
    @inbounds for j in 1:J-1
        for i in 1:I-1
            l,m = ϕ((i,j))
            #println("l-i = $(l-i), m-j = $(m-j)")
            dl = Int(floor(l))-range
            dm = Int(floor(m))-range
            ul = Int(ceil(l))+range
            um = Int(ceil(m))+range
            for a in max(1,dm):min(J-1,um)
                for b in max(1,dl):min(I-1,ul)
                    distance = sqrt(((l-b)^2)+((m-a)^2)) 
                    s = ker(distance)*wgt[b,a]*image[b,a]
                    #println("distance = $distance, s= $s,image[b,a] = $(image[b,a]), image[i,j] = $(image[i,j]) ")
                    imwarped[i,j] += s
                end
            end
        end
    end



    return imwarped
end

"""
To be done
"""
function OperatorConstruct(wgt::Matrix{Float64},Poly1::Poly2D,Poly2::Poly2D)

    ϕ(x) = (Poly1(x[1],x[2]),Poly2(x[1],x[2]))
    ker = CatmullRomSpline()    
    I = size(wgt)[1]
    J = size(wgt)[2]
    A = size(wgt)[1]
    B = size(wgt)[2]
    println(size(wgt))
    operator = zeros(I,J,A,B)
    range = 5
    for j in 1:J-1
        for i in 1:I-1
            l,m = ϕ((i,j))
            #println("l-i = $(l-i), m-j = $(m-j)")
            for b in 1:B-1
                for a in 1:A-1
                    distance = sqrt(((l-b)^2)+((m-a)^2)) 
                    s = ker(distance)*wgt[b,a]
                    #println("distance = $distance, s= $s,image[b,a] = $(image[b,a]), image[i,j] = $(image[i,j]) ")
                    operator[i,j,a,b] += s 
                end
            end
        end
    end



    return imwarped
end


"""
    function CreateTestSlitlet()


Return a fake slitlet filled with index of pixels
"""
function CreateTestSlitlet()

    #Actual SINFONI slitlet dimensions
    Dimx = 64
    Dimλ = 2048





    #Slitlets initialisation
    Slitlet = zeros(Float64,Dimλ,Dimx)


    #Filling slitlets

    for y in 1:Dimλ       
        for x in 1:Dimx
            Slitlet[y,x] = convert(Float64,x+(y-1)*Dimx)
        end  
    end



    return Slitlet

end



"""
    function CreateToySlitlet()

usage : julia> SlitletFibre, SlitletWave = CreateToySlitlet()

Return two fake SINFONI slitlet mimiquing FIBRE and WAVE slitlet
 filled with gaussian simulated signal and added zero margin for deformation space

"""
function CreateToySlitlet(gaussσ::Float64 = 1.)

    #Actual SINFONI slitlet dimensions
    Dimx = 64
    Dimλ = 2048

    #Slitlet's filling function
    σ = gaussσ
    gauss(x) = exp(-(x)^2/(2σ^2))/(σ*sqrt(2π))*1000

    #Slitlets initialisation
    SlitletFibre = zeros(Float64,Dimλ,Dimx)
    SlitletWave = zeros(Float64,Dimλ,Dimx)


    #Filling slitlets
    Centerx = Dimx ÷ 2 + 1/2
    for (x,col) in enumerate(eachcol(SlitletFibre))
        col .= gauss(x-Centerx)
    end

    RayNumber = 10
    for i in 1:RayNumber
        Centerλ = i*Dimλ/(RayNumber+1)
        for (λ,row) in enumerate(eachrow(SlitletWave))
            row .= row .+ gauss(λ-Centerλ)
        end
    end

    return SlitletFibre, SlitletWave

end


"""
    AddNoise!(image::Matrix{Float64},μ::Float64,σ::Float64)

Add normal noise to image

"""
function AddNoise!(image::Matrix{Float64},μ::Float64,σ::Float64)

    Dimλ = size(image)[1]
    Dimx = size(image)[2]


    #Noise generation
    noisedistrib = Normal(μ,σ)
    noise = rand(noisedistrib,(Dimλ,Dimx))

    #Adding noise
    image .+= noise
    
    return image
end

"""
    AddOutlier!(image::Matrix{Float64},freq::Float64,amp::Float64)

Add outlier to image, probabilitie of an outlier occuring in a pixel is based on a normal law

"""
function AddOutlier!(image::Matrix{Float64},freq::Float64,amp::Float64)

    Dimλ = size(image)[1]
    Dimx = size(image)[2]
    max = maximum(image)
    map = rand(Float64,(Dimλ,Dimx))
    mask = findall(map .> freq)
    map[mask] .= 0.
    map = map .* (max*amp)
    image .+= map

    return image
end

"""
    AddMargin(image::Matrix{Float64},marginv::Int64,marginh::Int64)

Add margin to an image

"""
function AddMargin(image::Matrix{Float64},marginv::Int64,marginh::Int64)
        
        sizev = size(image)[1]
        sizeh = size(image)[2]

        arrayv = zeros(Float64,sizev,marginh)
        arrayh = zeros(Float64,marginv,sizeh)
        corner = zeros(Float64,marginv,marginh)

        top = [corner arrayh corner]
        bottom = copy(top)

        middle = [arrayv image arrayv]

        result = [top;middle;bottom]

        return result
end



"""
    DeformSlitlet!(Slitlet, coeff = rand(Float64,(4,3)))

usage : DeformedSlitlet,coeff = DeformSlitlet!(OriginalSlitlet, [coeff = rand(Float64,(4,3))])

Return a deformed slitlet using circular permutation. And the coeficients used for deformation
"""
function DeformSlitlet!(Slitlet, coeff = rand(Float64,(4,3)))


    #Bending functions
    Bendλ(λ) = (λ^2)*coeff[1,1]*0 + (λ/100)*coeff[1,2] + 10coeff[1,3]
    Bendx(x) = (x^2)*coeff[3,1]*0 + (x/100)*coeff[3,2] + coeff[3,3]

    #Stretching functions
    Stretchλ(λ) = (λ^2)*coeff[2,1] + λ*coeff[2,2] + coeff[2,3]
    Stretchx(x) = (x^2)*coeff[3,1] + x*coeff[3,2] + coeff[3,3]

    #Bending horizontally
    for (x,col) in enumerate(eachcol(Slitlet))
        offset = convert(Int64,round(Bendx(x))) #Column offset in pixel
        circshift!(col, offset)
    end


    #Bending vertically
    for (λ,row) in enumerate(eachrow(Slitlet))
        offset = convert(Int64,round(Bendλ(λ))) #Column offset in pixel
        circshift!(row, offset)
    end


    return Slitlet,coeff
end

"""
    GaussFitRow(FakeFibre)

Levemberg Marquart fit of each row in a Fibre array.
Return a matrix {λx[ResultArray; σval σunc centerval centerunc]}
where σ is the stdev of each gaussian and center is the gaussian center
(unc values are associated uncertainties)

"""
function GaussFitRow(FakeFibre)
    #Slitlet's original filling function
    gauss(x,σ,center) = @. exp(-(x-center)^2/(2σ^2))/(σ*sqrt(2π))
    params = [5.,64.]
    ResultArray = Array{Float64}(undef, 0, 4)


    #Fitting horizontally
    for (λ,row) in enumerate(eachrow(FakeFibre))
        dom = Domain(1.:1:size(FakeFibre[λ,:])[1])
        model = Model(:comp => FuncWrap(gauss, params...))
        prepare!(model, dom, :comp)
        data = Measures(FakeFibre[λ,:],1.)
        result = fit!(model, data)
        σval = result.param[:comp__p1].val
        σunc = result.param[:comp__p1].unc
        centerval = result.param[:comp__p2].val
        centerunc = result.param[:comp__p2].unc
        ResultArray = [ResultArray; σval σunc centerval centerunc]

    end
    return ResultArray
end

"""
    GaussFit(ydata::Vector{Float64},p0::Vector{Float64} = [1., 4.])

Levemberg Marquart fit of a gaussian on an array.
Take data and initial parameters p0 = [σ,center] and return uptated parameters

"""
function GaussFit(ydata::Vector{Float64},p0::Vector{Float64} = [1., 4.])
    #Slitlet's original filling function
    model(t,p) = @. exp(-(t-p[2])^2/(2p[1]^2))/(p[1]*sqrt(2π)) #gaussian model
    tdata = 1:size(ydata)[1] #grid
    fit = curve_fit(model, tdata, ydata, p0) 
    param = fit.param

    return param
end
"""
    function PolynomFitRow(PolyArray)

Take the center of each row/column and return the polynomial least square fitting of it as a [a b c] matrix.
"""
function PolynomFitRow(PolyArray)

    filter!(x-> 0. <=x<= 128. ,PolyArray) #Filter out spurious values

    polynome(x,a,b,c) = @. (a*x^2+b*x+c)  #Polynomial to be fitted
    params = [0.,0.,0.]                   #Parameters of above polinomial [a,b,c]


    #Polynomial least square fitting
    dom = Domain(1.:1:size(PolyArray)[1])
    model = Model(:comp => FuncWrap(polynome, params...))
    prepare!(model, dom, :comp)
    data = Measures(PolyArray,10.)
    result = fit!(model, data)

    #Retrieving result
    a = result.param[:comp__p1].val
    b = result.param[:comp__p2].val
    c = result.param[:comp__p3].val
    return [a b c]
end


"""
    CenterVector(vector::vector{any})

Return the vector center's index (two indexes for even lenght)
"""
function CenterVector(vector::Vector)
    len = size(vector)[1]
    mid = convert(Int64,round((len+0.5)/2.))
    if isodd(len)
        return [mid]
    else 
        return [mid,mid+1]
    end
end

"""
    RayFinding(contrast::Vector{Float64},range::Int64,raynumber::Int64)

Return the raynumber number of gaussian peak index in contrast vector by flatenning contrast around each found peak in the range range.  
"""
function RayFinding(contrast::Vector{Float64},range::Int64,raynumber::Int64)


    contrastbis = copy(contrast)
    println()
    ind = zeros(Int64,raynumber)
    val = zeros(Float64,raynumber)

    for i in 1:raynumber
        ind[i] = findmax(contrastbis)[2]
        val[i] = findmax(contrastbis)[1]

        lowerbound = ind[i]-range
        upperbound = ind[i]+range
        if ind[i]-range < 1
            lowerbound = 1
        end
        if ind[i]+range > size(contrastbis)[1]
            upperbound = size(contrastbis)[1]
        end
        contrastbis[lowerbound:upperbound] .= 0.
    end
    return sort(ind)
end


"""
    RayFinding(contrast::Vector{Float32},range::Int64,raynumber::Int64)

Return the raynumber number of gaussian peak index in contrast vector by flatenning contrast around each found peak in the range range.  
"""
function RayFinding(contrast::Vector{Float32},range::Int64,raynumber::Int64)


    contrastbis = copy(contrast)
    println()
    ind = zeros(Int64,raynumber)
    val = zeros(Float64,raynumber)

    for i in 1:raynumber
        ind[i] = findmax(contrastbis)[2]
        val[i] = findmax(contrastbis)[1]

        lowerbound = ind[i]-range
        upperbound = ind[i]+range
        if ind[i]-range < 1
            lowerbound = 1
        end
        if ind[i]+range > size(contrastbis)[1]
            upperbound = size(contrastbis)[1]
        end
        contrastbis[lowerbound:upperbound] .= 0.
    end
    return sort(ind)
end

"""
    ReformSlitlet(coefvector::Vector{Float64},order::Int64,refpix::Tuple{Int,Int},imageref::Matrix{Float64},imageref2::Matrix{Float64})

Reform slitlet and store residual
"""
function ReformSlitlet(coefvector::Vector{Float64},order::Int64,refpix::Tuple{Int,Int},image::Matrix{Float64},image2::Matrix{Float64},imageref::Matrix{Float64},imageref2::Matrix{Float64})
    
    #Reformating input
    coefmatrix = reshape(coefvector,:,2)


    #Creating polynomials
    Poly1 = Poly2D(order,refpix,coefmatrix[:,1])
    Poly2 = Poly2D(order,refpix,coefmatrix[:,2])

    #Deforming image
    axess = axes(image)
    deformedimage = ImageWarp(image,Poly1,Poly2,axess)
    deformedimage2 = ImageWarp(image2,Poly1,Poly2,axess)

    #Calculating contrast
    contrast = vec(Contrast(deformedimage,1))  
    contrast2 = vec(Contrast(deformedimage2,2))

    f = FITS("ToyModelImages/finalfibre.fits", "w")
    write(f,deformedimage)
    close(f)
    f = FITS("ToyModelImages/finalwave.fits", "w")
    write(f,deformedimage2)
    close(f)


    #debug and IO
    if typeof(imageref) == typeof(deformedimage)
        f = FITS("ToyModelImages/finalresfibre.fits", "w")
        write(f,Residual(imageref,deformedimage))
        close(f)
        f = FITS("ToyModelImages/finalcontrastfibre.fits", "w")
        write(f,contrast)
        close(f)
    end
    if typeof(imageref2) == typeof(deformedimage2)
        f = FITS("ToyModelImages/finalreswave.fits", "w")
        write(f,Residual(imageref2,deformedimage2))
        close(f)
        f = FITS("ToyModelImages/finalcontrastwave.fits", "w")
        write(f,contrast2)
        close(f)
    end
    return deformedimage, deformedimage2
end


"""
    MinCriteria(image::Matrix,order::Int,refpix::Tuple{Int,Int})

Find the polynomials allowing image to be rectified
"""
function MinCriteria(image::Matrix{Float64},image2::Matrix{Float64},wgt::Matrix{Float64},order::Int64,refpix::Tuple{Int,Int},imageref::Matrix{Float64},imageref2::Matrix{Float64};fitswriting = false)

    peaknumber = 10
    iterator = 0

    #Get info once for all
    dim1λ = size(image)[1]
    dim1x = size(image)[2]
    dim2λ = size(image)[1]
    dim2x = size(image)[2]


    #Reference contrast
    contrastref = vec(Contrast(imageref2,2))
    axess = axes(image)
    centerwave = RayFinding(contrastref,10,peaknumber)

    #Total signal
    sumfibre = sum(image)
    sumwave = sum(image2)

    #Form identity polynomials and coefiscient
    polyidx =  Poly2D(order,refpix)
    polyidy =  Poly2D(order,refpix)
    CoefConstraintIdentity!(polyidx,1)
    CoefConstraintIdentity!(polyidy,2)
    coefidentity1 = vec(convert(Vector{Float64},copy(polyidx.coef)))
    coefidentity2 = vec(convert(Vector{Float64},copy(polyidy.coef)))
    coefidentity = vec(convert(Vector{Float64},[polyidx.coef; polyidy.coef]))
    genmap = polyidx.map

    #Preform deformation polynomials ( so that only the coefiscients are to be updated afterward)
    Poly1 = Poly2D(order,refpix)
    Poly2 = Poly2D(order,refpix)
 

    #Poping zeroth order and undesirable degrees of freedom
    #pop!(coefidentity1)
    #pop!(coefidentity2)
    #pop!(coefidentity1)
    #pop!(coefidentity2)
    #coefidentity1 .= 0. 
    #coefidentity1 .= 0. 

       
    #Minimisation characteristics (box's limit, starting point and algorithm)   
    coefmapsum = append!(vec(polyidx.map[1] .+ polyidx.map[2]),vec(polyidy.map[1] .+ polyidy.map[2]))
    bounds = Array{Tuple{Float64, Float64}}(undef, 0)
    for i in 1:size(coefmapsum)[1]
        limit = 1. *10. ^(-coefmapsum[i]*2)
        if i <= size(coefmapsum)[1]/2
            limit *= 10.
        end
        push!(bounds,(coefidentity[i]-limit,coefidentity[i]+limit))
    end
    println("bounds = $bounds")

    #Minimisation parameters
    initial_x = coefidentity
    precisionfibre = 1e-1
    precisionwave = precisionfibre*1e-5
    inner_optimizer = NelderMead()


    log = open("ToyModelImages/log.txt", "a")
    write(log, "
    Minimisation :

    initial_x = $initial_x
    g_tolfibre = $precisionfibre
    g_tolwave = $precisionwave
    inner_optimizer = $inner_optimizer")
    close(log)

    """
    #aggregator
    weightedfitness(y) = y[1]/size(image)[1] + y[2]/(size(image)[2]*peaknumber)
    """
    """
    MinCoupled(coefvector::Vector{Float64})

Function used for coupled minimisation of fibre and wave frames
"""
function MinCoupled(coefvector::Vector{Float64})

    #Bounding
    for i in 1:size(coefvector)[1]
        if !(bounds[i][1] < coefvector[i] < bounds[i][2]) 
            println("Out Of Bounds : $(bounds[i][1]) < $(coefvector[i]) < $(bounds[i][2])")
            return 0.
        end
    end
    

    """
    #allocating memory
    deformedimage = Array{Float64}(undef, dim1λ, dim1x)
    contrast = Array{Float64}(undef, dim1x)
    deformedimage2 = Array{Float64}(undef, dim2λ, dim2x)
    contrast2 = Array{Float64}(undef, dim2λ)
    """

    #Reformating input
    coefmatrix = reshape(coefvector,:,2)


    #Creating polynomials
    Poly1 = Poly2D(order,refpix,vec(coefmatrix[:,1]),genmap)
    Poly2 = Poly2D(order,refpix,vec(coefmatrix[:,2]),genmap)




    #Deforming image
    deformedimage = ImageWarp(image,Poly1,Poly2,axess)
    deformedimage2 = ImageWarp(image2,Poly1,Poly2,axess)


    #Calculating criteria of minimisation

    sumdeformedfibre = sum(deformedimage)
    sumdeformedwave = sum(deformedimage)
 

    uprow = convert(Int64,(size(deformedimage)[1]-2048)/2)
    downrow  = convert(Int64,size(deformedimage)[1]-uprow)
    upcol = convert(Int64,(size(deformedimage)[2]-64)/2)
    downcol = convert(Int64,size(deformedimage)[2]-upcol)
    println("uprow = $uprow downrow = $downrow upcol = $upcol downcol = $downcol")


    contrast = vec(Contrast(deformedimage[uprow:downrow-1,upcol:downcol-1],1))
    center = CenterVector(contrast)
    sumgauss = sum(contrast[center .+ 1])
    stdev = std(vec(Contrast(deformedimage[uprow:downrow-1,upcol:downcol-1],2)))
    criteria1 = -sumgauss#/stdev

    contrast2 = vec(Contrast(deformedimage2[uprow:downrow-1,upcol:downcol-1],2))
    sumgauss2 = sum(contrast2[centerwave .- uprow .+ 1])
    stdevwave = std(vec(contrast2[centerwave .- uprow .+ 1]))
    centerdef = RayFinding(contrast2,10,peaknumber)
    centerdiff = sum(abs.((centerdef .^ 2) .- (centerwave .^ 2)))
    stdev2 = std(vec(Contrast(deformedimage2[uprow:downrow-1,upcol:downcol-1],1)))
    criteria2 = -sumgauss2#/stdev2
    println("center = $center,corr = $(center .- uprow),imsize = $(size(deformedimage)), contsize = $(size(contrast)),contsize2 = $(size(contrast2)), centerwave = $centerwave, centerwavecorr = $(centerwave .- uprow) ")

    weight = (sumfibre/sumwave)^2

    criteriasum = (criteria1 + criteria2*weight)#/centerdiff
    println("centerdiff = $centerdiff")

    #debug and IO
    iterator += 1
    if iterator%10 == 0
        if fitswriting == true
            if typeof(imageref) == typeof(deformedimage)
                f = FITS("ToyModelImages/fcrit=$criteria1.fits", "w")
                write(f,Residual(imageref,deformedimage))
                close(f)
                f = FITS("ToyModelImages/sumdeformedfibre=$sumdeformedfibre.fits", "w")
                write(f,deformedimage)
                close(f)
                f = FITS("ToyModelImages/fcontrastcrit=$criteria1.fits", "w")
                write(f,contrast)
                close(f)
            end
            if typeof(imageref2) == typeof(deformedimage2)
                f = FITS("ToyModelImages/wcrit=$criteria2.fits", "w")
                write(f,Residual(imageref2,deformedimage2))
                close(f)
                f = FITS("ToyModelImages/sumdeformedwave=$sumdeformedwave.fits", "w")
                write(f,deformedimage2)
                close(f)
                f = FITS("ToyModelImages/wcontrastcrit=$criteria2.fits", "w")
                write(f,contrast2)
                close(f)
            end
        end
    end

    println("coef1 = $(Poly1.coef), coef2 = $(Poly2.coef), criteria1 = $criteria1, criteria2 = $criteria2 ,weight = $weight, criteriasum = $criteriasum")

    return criteriasum
end
    
    """
        Min(coefvector::Vector{Float64})
    
    Function used for minimisation
    """
    function Min(coefvector::Vector{Float64})

        #Reformating input
        coefmatrix = reshape(coefvector,:,2)

        #Creating polynomials
        Poly1 = Poly2D(order,refpix,coefmatrix[:,1])
        Poly2 = Poly2D(order,refpix,coefmatrix[:,2])

        #Deforming image
        axess = axes(image)
        deformedimage = ImageWarp(image,Poly1,Poly2,axess)
        deformedimage2 = ImageWarp(image2,Poly1,Poly2,axess)

        #Calculating criteria of minimisation

        contrast = vec(Contrast(deformedimage,1))
        center = CenterVector(contrast)
        criteria1 = -sum(contrast[center])
      
        contrast2 = vec(Contrast(deformedimage2,2))
        contrastref = vec(Contrast(imageref2,2))
        center2 = RayFinding(contrastref,10,peaknumber)
        criteria2 = -sum(contrast2[center2])

        criteria = criteria1 + criteria2*((sum(image)/(sum(image2)))^2)

        #criteria = criteria1/size(deformedimage)[1] + (2048/640)*criteria2/(size(deformedimage2)[2]*peaknumber)
        iterator += 1
        #debug and IO
        println("criteria=$criteria
        size(deformedimage)[1]= $(size(deformedimage)[1])
        (size(deformedimage2)[2]*peaknumber) = $((size(deformedimage2)[2]*peaknumber))
        ratioline = $((size(deformedimage)[1])/(size(deformedimage2)[2]*peaknumber))
        ratiosumdef = $(sum(deformedimage)/sum(deformedimage2))
        ratiosumim = $(sum(image)/sum(image2))")
        if iterator%100 == 0
            if fitswriting == true
                if typeof(imageref) == typeof(deformedimage)
                    f = FITS("ToyModelImages/crit=$criteria1.fits", "w")
                    write(f,Residual(imageref,deformedimage))
                    close(f)
                    f = FITS("ToyModelImages/contrastcrit=$criteria1.fits", "w")
                    write(f,contrast)
                    close(f)
                end
                if typeof(imageref2) == typeof(deformedimage2)
                    f = FITS("ToyModelImages/crit2=$criteria2.fits", "w")
                    write(f,Residual(imageref2,deformedimage2))
                    close(f)
                    f = FITS("ToyModelImages/contrastcrit2=$criteria2.fits", "w")
                    write(f,contrast2)
                    close(f)
                end
            end
        end
        return criteria
    end

    """
        Min2(coefvector::Vector{Float64})
    
    Alternative Function used for minimisation
    """
    function Min2(coefvector::Vector{Float64})


        #Reformating input
        coefmatrix = reshape(coefvector,:,2)

        #Creating polynomials
        Poly1 = Poly2D(order,refpix,coefmatrix[:,1])
        Poly2 = Poly2D(order,refpix,coefmatrix[:,2])

        #Deforming image
        axess = axes(image)
        deformedimage = ImageWarp(image,Poly1,Poly2,axess)
        deformedimage2 = ImageWarp(image2,Poly1,Poly2,axess)

        #Calculating criteria of minimisation

        contrast = vec(Contrast(deformedimage,1))
        center = CenterVector(contrast)
        criteria1 = -sum(contrast[center])

        contrast2 = vec(Contrast(deformedimage2,2))
        contrastref = vec(Contrast(imageref2,2))
        center2 = RayFinding(contrastref,10,peaknumber)
        criteria2 = -sum(contrast2[center2])


        #criteria = std(Contrast(deformedimage,2))*10-std(Contrast(deformedimage,1))-(sum(deformedimage)/(size(deformedimage)[1]*size(deformedimage)[2]))

        #debug and IO
        println("coef1=$(Poly1.coef)")
        println("coef2=$(Poly2.coef)")
        if fistwriting == true
            if typeof(imageref) == typeof(deformedimage)
                f = FITS("ToyModelImages/crit=$criteria1.fits", "w")
                write(f,Residual(imageref,deformedimage))
                close(f)
                f = FITS("ToyModelImages/contrastcrit=$criteria1.fits", "w")
                write(f,contrast)
                close(f)
            end
            if typeof(imageref2) == typeof(deformedimage2)
                f = FITS("ToyModelImages/crit2=$criteria2.fits", "w")
                write(f,Residual(imageref2,deformedimage2))
                close(f)
                f = FITS("ToyModelImages/contrastcrit2=$criteria2.fits", "w")
                write(f,contrast2)
                close(f)
            end
        end
        return (criteria1,criteria2)
    end

    """
    MinFibre(coefvector::Vector{Float64})

Function used for separate minimisation of a fibre frame
"""
function MinFibre(coefvector::Vector{Float64})

    #allocating memory
    deformedimage = Array{Float64}(undef, dim1λ, dim1x)
    contrast = Array{Float64}(undef, dim1x)

    #Reformating input
    coefvectorcopy = copy(coefvector)

    #Pushing zeroth order
    #push!(coefvectorcopy,refpix[1])
    #insert!(coefvectorcopy,size(coefvectorcopy)[1],1.)

    


    #Creating polynomials
    Poly1 = polyidx
    #Poly2 = Poly2D(order,refpix,vec(coefvectorcopy),genmap)
    Poly2.coef =  vec(coefvectorcopy)




    #Deforming image
    deformedimage = ImageWarp2(image,wgt,Poly1,Poly2)
    #Calculating criteria of minimisation


    sumdeformedfibre = sum(deformedimage)
    
    sumdeformedfibre2 = 0.
    if sumdeformedfibre < 0.
        for pix in deformedimage
            sumdeformedfibre2 += pix
            if pix < 0.
                #println(pix)
            end
        end 
    end
    
    #println("sumfibre = $sumfibre, sumdeformedfibre = $sumdeformedfibre, sumdeformedfibre2 = $sumdeformedfibre2")
    println("fcoef1 = $(Poly1.coef), fcoef2 = $(Poly2.coef)")

    contrast = vec(Contrast(deformedimage,1))
    center = CenterVector(contrast)
    sumgauss = sum(contrast[center])
    sumrest = sum(contrast) - sum(contrast[center])
    stdev = std(vec(Contrast(deformedimage,2)))
    criteria = -sumgauss/stdev#+sumrest
    #println("center = $center")


    iterator += 1
    #debug and IO
    if sumdeformedfibre < 0.
        if fitswriting == true
            if typeof(imageref) == typeof(deformedimage)
                f = FITS("ToyModelImages/fcrit=$criteria.fits", "w")
                write(f,Residual(imageref,deformedimage))
                close(f)
                f = FITS("ToyModelImages/sumdeformedfibre=$sumdeformedfibre.fits", "w")
                write(f,deformedimage)
                close(f)
                f = FITS("ToyModelImages/fcontrastcrit=$criteria.fits", "w")
                write(f,contrast)
                close(f)
            end
        end
    end

    return criteria
end

"""
MinWave(coefvector::Vector{Float64})

Function used for separate minimisation of a wave frame
"""
function MinWave(coefvector::Vector{Float64})

    #allocating memory
    deformedimage = Array{Float64}(undef, dim2λ, dim2x)
    contrast = Array{Float64}(undef, dim2λ)


    #Reformating input
    coefvectorcopy = copy(coefvector)

    #pushing zeroth order
    #push!(coefvectorcopy,refpix[2])
    #insert!(coefvectorcopy,size(coefvectorcopy)[1]-1,1.)


    #Creating polynomials
    #Poly1 = Poly2D(order,refpix,vec(coefvectorcopy),genmap)
    Poly1.coef =  vec(coefvectorcopy)
    Poly2 = polyidy


    sumdeformedwave = sum(deformedimage)
    #println("sumwave = $sumwave, sumdeformedwave = $sumdeformedwave")
    println("wcoef1 = $(Poly1.coef), wcoef2 = $(Poly2.coef)")


    #Deforming image
    deformedimage = ImageWarp2(image2,wgt,Poly1,Poly2)
    #Calculating criteria of minimisation

    contrast = vec(Contrast(deformedimage,2))
    #println(centerwave)
    sumgauss = sum(contrast[centerwave])
    sumrest = sum(contrast) - sum(contrast[centerwave])
    stdev = std(vec(Contrast(deformedimage,1)))
    criteria = -sumgauss/stdev#+sumrest
    #println("centerwave = $centerwave")


    #debug and IO
    iterator += 1
    if iterator%1 == 0
        if fitswriting == true
            if typeof(imageref2) == typeof(deformedimage)
                f = FITS("ToyModelImages/wcrit=$criteria.fits", "w")
                write(f,Residual(imageref2,deformedimage))
                close(f)
                f = FITS("ToyModelImages/sumdeformedwave=$sumdeformedwave.fits", "w")
                write(f,deformedimage)
                close(f)
                f = FITS("ToyModelImages/wcontrastcrit=$criteria.fits", "w")
                write(f,contrast)
                close(f)
            end
        end
    end

    return criteria
end


"""
    #ForwardDiff
    MinFibre! = x::Vector{Float64} -> ForwardDiff.gradient(MinFibre, x)
    MinWave! = x::Vector{Float64} -> ForwardDiff.gradient(MinWave, x)
    println(coefidentity2)
    prinln(MinWave!(coefidentity1))
"""

    println("initial1 = $coefidentity1, initial2 = $coefidentity2")
    #Minimisation
    res1 = optimize(MinFibre, coefidentity2, inner_optimizer, Optim.Options(g_tol = precisionfibre, store_trace = false, show_trace = false))
    res2 = optimize(MinWave, coefidentity1, inner_optimizer, Optim.Options(g_tol = precisionwave, store_trace = false, show_trace = false))
    res = [res1;res2]

    #res = optimize(MinCoupled, [coefidentity1;coefidentity2], inner_optimizer, Optim.Options(g_tol = precisionfibre, store_trace = false, show_trace = false, iterations = 10000))

    """
    res = bboptimize(Min2,initial_x; Method=:borg_moea,
    FitnessScheme=ParetoFitnessScheme{2}(is_minimizing=true,aggregator=weightedfitness),
    SearchRange=bounds, NumDimensions=size(initial_x)[1], ϵ=precision,
    MaxSteps=50000, TraceInterval=1.0, TraceMode=:verbose);
    @assert best_fitness(res) == fitness_2obj(initial_x)
    """
    #res = optimize(Min, initial_x, inner_optimizer, Optim.Options(g_tol = precision, store_trace = true, show_trace = true))
    return res
end

"""
    MultAxes(x::Int64,img::Matrix{Float64})
    
Return the axes base.OneTo value of an array the size of x time img in every direction
"""
function MultAxes(x::Int64,img::Matrix{Float64})
    y = zeros(size(img) .* x)
    return axes(y)
end

"""
    ImgFilter(img::Matrix{Float64},uptresh::Float64,downtresh::Float64)
    
Return the image filtered from points having intensity outsides μ ± σ*boundaries
"""
function ImgFilter(img::Matrix{Float64},uptresh::Float64,downtresh::Float64)

    image = copy(img)

    σ = std(image)
    μ = mean(image)
    min = minimum(image)
    max = maximum(image)

    uptresh = max - σ*uptresh
    downtresh = min + σ*downtresh

    mask1 = findall(image .> uptresh)
    mask2 = findall(image .< downtresh)

    image[[mask1;mask2]] .= 0.


    return im 
end


"""
    ImgFilter!(img::Matrix{Float64},uptresh::Float64,downtresh::Float64)
    
Return the image filtered from points having intensity outsides μ ± σ*boundaries
"""
function ImgFilter!(img::Matrix{Float64},uptresh::Float64,downtresh::Float64)

    σ = std(img)
    μ = mean(img)

    uptresh = μ + σ*uptresh
    downtresh = μ - σ*downtresh

    mask1 = findall(img .> uptresh)
    mask2 = findall(img .< downtresh)

    img[[mask1;mask2]] .= 0.


    return img 
end




"""
    oldmain()

Obsolete Toy Model
"""
function oldmain()

    FakeFibre, FakeWave = CreateToySlitlet() #Create the slitlets

    #Deform the image with circular permutation and random coeficient
    FakeWaveDist,coeffs  = DeformSlitlet(FakeWave)

    #Deform the image with circular permutation and the previous coeficient
    FakeFibreDist,coeffs  = DeformSlitlet(FakeFibre,coeffs)

    CenterPos = GaussFitRow(FakeFibreDist) #Return an array of the position of each gaussian center

    Polycoef = PolynomFitRow(PolyArray) #Return the coeficient of the polynomial associated

end





"""
    main2()

Toy Model (detector to sky version)
"""
function main2()

    timestart = time()

    gaussσ = 1.


    FakeFibre, FakeWave = CreateToySlitlet(gaussσ) #Create the slitlets
    #FakeFibre, FakeWave = CreateToySlitlet(1.,0.,0.) #Create the slitlets

    noiseμ = maximum(FakeFibre)/10.
    noiseσ = noiseμ/10.
    outlier_frec = 0.01
    outlier_amp= 100.
    order = 2
    axemult = 1


    
    rm("ToyModelImages",force=true,recursive = true)
    mkdir("ToyModelImages")
    f = FITS("ToyModelImages/ContrastFibre.fits", "w")
    write(f,Contrast(FakeFibre,1))
    close(f)
    f = FITS("ToyModelImages/ContrastWave.fits", "w")
    write(f,Contrast(FakeWave,2))
    close(f)
    f = FITS("ToyModelImages/FakeFibre.fits", "w")
    write(f,FakeFibre)
    close(f)
    f = FITS("ToyModelImages/FakeWave.fits", "w")
    write(f,FakeWave)
    close(f)


    #Generate 2 2DPolynomials with defined coefiscients

    if order == 2
        refpix = (0,0)
        coef1 = [0.,0.,0.,1.005,0.05,0.]
        coef2 = [0.,0.,0.0001,0.008,1.,0.]
    end

    if order == 1
        refpix = (0,0)
        coef1 = [1.,0.05,0.]
        coef2 = [0.008,1.,0.]
    end

    #Adding noise
    AddNoise!(FakeFibre,noiseμ,noiseσ)
    AddNoise!(FakeWave,noiseμ,noiseσ)

    """
    #Adding outliers
    AddOutlier!(DeformedFakeFibre,outlier_frec,outlier_amp)
    AddOutlier!(DeformedFakeWave,outlier_frec,outlier_amp)
    """

    wgtmap = ones(size(FakeFibre)[1],size(FakeFibre)[2])

    Poly1 = Poly2D(order,refpix,coef1)
    Poly2 = Poly2D(order,refpix,coef2)

    #Adding margin
    λmax = size(FakeFibre)[1]*1.
    xmax = size(FakeFibre)[2]*1.

    marginλ = convert(Int64,round(Poly1(λmax,xmax)- λmax+1.)) 
    marginx = convert(Int64,round(Poly2(λmax,xmax)- xmax+1.))

    FakeFibre = AddMargin(FakeFibre,marginλ,marginx)
    FakeWave= AddMargin(FakeWave,marginλ,marginx)



    #generate axes for warp transformation
    axess = MultAxes(axemult,FakeFibre)
    
    #Deform the images with ImageTransformations.warp 
    DeformedFakeFibre = ImageWarp(FakeFibre,Poly1,Poly2,axess)
    DeformedFakeWave = ImageWarp(FakeWave,Poly1,Poly2,axess)

    f = FITS("ToyModelImages/DeformedFakeFibreNoNoise.fits", "w")
    write(f,DeformedFakeFibre)
    close(f)
    f = FITS("ToyModelImages/DeformedFakeWaveNoNoise.fits", "w")
    write(f,DeformedFakeWave)
    close(f)

    


    """
    #Filtering hot pixels and noise 
    DeformedFakeFibre = ImgFilter(DeformedFakeFibreUnfiltered,3.,3.)
    DeformedFakeWave = ImgFilter(DeformedFakeWaveUnfiltered,3.,3.)
    """




    log = open("ToyModelImages/log.txt", "a")
    logtext = "Polynomials :

    refpix = $refpix
    order = $order
    gaussσ = $gaussσ
    noiseμ = $noiseμ
    noiseσ = $noiseσ
    outlier_frec = $outlier_frec
    outlier_amp = $outlier_amp
    marginλ = $marginλ
    marginx = $marginx
    axemult = $axemult
       
    coef1 = $coef1
    map1 = $(Poly1.map)

    coef2 = $coef2
    map2 = $(Poly2.map)"
    println(log,logtext)
    close(log)

    
    
    f = FITS("ToyModelImages/DeformedFakeFibre.fits", "w")
    write(f,DeformedFakeFibre)
    close(f)
    f = FITS("ToyModelImages/ResDeformedFakeFibre.fits", "w")
    write(f,Residual(FakeFibre,DeformedFakeFibre))
    close(f)
    f = FITS("ToyModelImages/DeformedFakeWave.fits", "w")
    write(f,DeformedFakeWave)
    close(f)
    f = FITS("ToyModelImages/ResDeformedFakeWave.fits", "w")
    write(f,Residual(FakeWave,DeformedFakeWave))
    close(f)

    """
    coef3 = [0.985429,0.0180358,0.0182765]
    coef4 = [-0.00981042,0.523253,0.00426099]
    Poly3 = Poly2D(order,refpix,coef3)
    Poly4 = Poly2D(order,refpix,coef4)
    ReformedFakeFibre = ImageWarp(DeformedFakeFibre,Poly3,Poly4,axess)
    f = FITS("ToyModelImages/ReformedFakeFibre.fits", "w")
    write(f,ReformedFakeFibre)
    close(f)
    """


    minimisation = MinCriteria(DeformedFakeFibre,DeformedFakeWave,wgtmap,order,refpix,FakeFibre,FakeWave;fitswriting = true)
    
    #Reinserting constant 
    #push!(Optim.minimizer(minimisation[1]),1.)
    #push!(Optim.minimizer(minimisation[1]),refpix[1])

    #insert!(Optim.minimizer(minimisation[2]),size(Optim.minimizer(minimisation[2]))[1],1.)
    #push!(Optim.minimizer(minimisation[2]),refpix[2])

    log = open("ToyModelImages/log.txt", "a")
    logtext = "
    
    Minimisation :

    $(minimisation)

    
    minimizer :
    
    $(Optim.minimizer(minimisation))
    
    minimimum :
    
    $(Optim.minimum(minimisation))
   
    wallclock duration : $(time() - timestart) seconds"

    println(log,logtext)
    close(log)


    reformcoef = vec(Optim.minimizer(minimisation))

    
    """
    log = open("ToyModelImages/log.txt", "a")
    logtext = "
    
    Minimisationfibre :

    (minimisation[1])

    Minimisationwave :

    (minimisation[2])
    
    minimizer :
    
    (Optim.minimizer(minimisation[2]))
    (Optim.minimizer(minimisation[1]))
    
    minimimum :
    
    (Optim.minimum(minimisation[2]))
    (Optim.minimum(minimisation[1]))
    
    wallclock duration : (time() - timestart) seconds"

    println(log,logtext)
    close(log)


    reformcoef = vec([Optim.minimizer(minimisation[2]) Optim.minimizer(minimisation[1])])
    """

    
    finaldeformedfibre, finaldeformedwave = ReformSlitlet(reformcoef,order,refpix,DeformedFakeFibre,DeformedFakeWave,FakeFibre,FakeWave)

    errorfibre = sum(Residual(FakeFibre,finaldeformedfibre).^2)/sum(Residual(FakeFibre,DeformedFakeFibre).^2)
    errorwave = sum(Residual(FakeWave,finaldeformedwave).^2)/sum(Residual(FakeWave,DeformedFakeWave).^2)
    log = open("ToyModelImages/log.txt", "a")
    logtext = "
    
    errorfibre = $errorfibre
    errorwave = $errorwave"

    println(log,logtext)
    println(logtext)
    close(log)


    return minimisation


end


"""
    function projection(::Val{:proj},
        pos::AbstractVector, # output positions
        wgt::AbstractMatrix, # weights of data
        dat::AbstractMatrix, # data
        ker,         # interpolation kernel
        coord;       # coordinate function called as coord(i,j)
        support=10)

Example:
    p = projection(Val(:proj), 1:200, wgt, img, CatmullRomSpline{Float64}(),
                   (i,j) -> 3.0*i - 2.5 + 0.01*j)
    I = find_peaks(p; nmax=30)
"""
function projection(::Val{:proj},
                    pos::AbstractVector, # output positions
                    wgt::AbstractMatrix, # weights of data
                    dat::AbstractMatrix, # data
                    ker,         # interpolation kernel
                    coord;       # coordinate function called as coord(i,j)
                    support=10)
    axes(wgt) == axes(dat) || error("incompatible indices")
    I, J = axes(dat)
    len = length(pos)
    A = zeros(Float64, len)
    B = zeros(Float64, len)
    axes(pos) == (1:len,) || error("non-standard indices not supported")
    @inbounds for j in J
        for i in I
            x = coord(i,j)
#            for k in 1:len
#            for k in max(1,Int(floor(x))-10):min(len,Int(ceil(x))+10)
            pfirst = searchsortedfirst(pos,x-support)
            plast = searchsortedlast(pos,x+support)
            for (k,p) in enumerate(pos[pfirst:plast])
                w = ker(x - p)*wgt[i,j]
                A[k+pfirst-1] += w
                B[k+pfirst-1] += w*dat[i,j]
            end
        end
    end
    @inbounds for k in eachindex(A, B)
        if A[k] != 0
            B[k] /= A[k]
        else
            B[k] = 0
        end
    end
    return B
end

"""
    projection(::Val{:matched},
                        pos::AbstractVector, # output positions
                        wgt::AbstractMatrix, # weights of data
                        dat::AbstractMatrix, # data
                        ker,         # footprint
                        coord)       # coordinate function called as coord(i,j)
"""
function projection(::Val{:matched},
                    pos::AbstractVector, # output positions
                    wgt::AbstractMatrix, # weights of data
                    dat::AbstractMatrix, # data
                    ker,         # footprint
                    coord)       # coordinate function called as coord(i,j)
    axes(wgt) == axes(dat) || error("incompatible indices")
    I, J = axes(dat)
    len = length(pos)
    A = zeros(Float64, len)
    B = zeros(Float64, len)
    axes(pos) == (1:len,) || error("non-standard indices not supported")
    @inbounds for j in J
        for i in I
            x = coord(i,j)
            for k in 1:len
                phi = ker(x - pos[k])
                w = wgt[i,j]*phi
                A[k] += w*phi
                B[k] += w*dat[i,j]
            end
        end
    end
    @inbounds for k in eachindex(A, B)
        if A[k] != 0
            B[k] /= sqrt(A[k])
        else
            B[k] = 0
        end
    end
    return B
end

"""
    find_peaks!(vect::AbstractVector{<:Real};
                        dist::Integer = 5,
                        rtol::Real = 0.1,
                        atol::Real = 0.0,
                        nmax::Integer = typemax(Int))

    find_peaks!(vect) -> inds
yields the indices of the most significant local maxima found in vector `vect`
destroying the contents of `vect` in the process.  The indices are returned in
decreasing order of peak heights.
The following keywords can be used to tune the algorithm:
* `dist` (5 by default) specifies the minimal distance (in indices units)
  between two peaks.
* `atol` (0.0 by default) and `rtol` (0.1 by default) specify the absolute and
  relative tolerances for the detection threshold.  If `atol` is NaN, the
  detection threshold is `rtol*maximum(vect)`; otherwise, the detection
  threshold is `max(atol, rtol*maximum(vect))`.  All selected peaks have values
  greater or equalt the detection threshold.
* `nmax` (no limits by default) specifies the maximum number of peaks to
  detect.
Call `find_peaks` to avoid overwriting `vect`.
"""
function find_peaks!(vect::AbstractVector{<:Real};
                     dist::Integer = 5,
                     rtol::Real = 0.1,
                     atol::Real = 0.0,
                     nmax::Integer = typemax(Int))
    dist ≥ 1 || error("minimal distance must be ≥ 1")
    0 ≤ rtol ≤ 1 || error("invalid relative threshold")

    dst = Int(dist) - 1
    vmin = typemin(eltype(vect))
    vtol = float(atol)
    I = axes(vect, 1)
    I_first, I_last = first(I), last(I)
    inds = Int[]
    while length(inds) < nmax
        vmax, imax = findmax(vect)
        if length(inds) < 1
            # Compute selection threshold.
            v = oftype(vtol, rtol*vmax)
            if isnan(vtol)
                vtol = v
            else
                vtol = max(vtol, v)
            end
        end
        vmax ≥ vtol || break
        push!(inds, imax)
        @inbounds for i in max(I_first, imax - dst):min(I_last, imax + dst)
            vect[i] = vmin
        end
    end
    return inds
end

"""
    find_peaks(vect) -> inds
yields the indices of the most significant local maxima found in vector `vect`.
See `find_peaks!` for a list of accepted keywords.
"""
find_peaks(vect::AbstractVector, args...; kwds...) =
    find_peaks!(copy(vect), args...; kwds...)


function deprojection(proj::AbstractVector, # projection
                    imgAxes::Tuple{AbstractRange,AbstractRange},
                    ker,         # interpolation kernel
                    coord;       # coordinate function called as coord(i,j)
                    support=10)
   deprojection(proj,imgAxes,ker,coord,1:length(proj); support=support)
end

function deprojection(proj::AbstractVector, # projection
                    imgAxes::Tuple{AbstractRange,AbstractRange},
                    ker,         # interpolation kernel
                    coord,       # coordinate function called as coord(i,j)
                    indices::AbstractRange;
                    support=10)
    I, J = imgAxes
    len = length(proj)
    img = zeros(imgAxes)
    axes(indices) == axes(proj) || error("incompatible indices")
    @inbounds for j in J
        for i in I
            x = coord(i,j)
            pfirst = searchsortedfirst(indices,x-support)
            plast = searchsortedlast(indices,x+support)

            for (k,p) in enumerate(indices[pfirst:plast])
                img[i,j] += ker(x -p )*proj[k + pfirst-1]
            end
        end
    end
    return img
end


"""
    GaussianModel(fwhm::Float64, x::Real)
Compute the value at position sqrt(x) 1D centered Gaussian
* `fwhm` : full-width at half maximum
* `x`:   sqrt sampled position
"""
function GaussianModel(fwhm, x::T) where (T<:Real)
    local fwhm2sigma =T(1 / (2 * sqrt(2 * log( 2))))*T(fwhm)
    return exp(-x / (2 * (fwhm2sigma )^2));
end

"""
    GaussianModel2(fwhm::Float64, x::Real)
Compute the value at position x 1D centered Gaussian
* `fwhm` : full-width at half maximum
* `x`:   sampled position
"""
function GaussianModel2(fwhm, x::T) where (T<:Real)
    local fwhm2sigma =T(1 / (2 * sqrt(2 * log( 2))))*T(fwhm)
    return exp(-x^2 / (2 * (fwhm2sigma )^2));
end


"""
    SlitletLoad(fitsname,directory,xmin::Int64,xmax::Int64,ymin::Int64,ymax::Int64)

Load a slitlet from disk
"""

function SlitletLoad(fitsname,directory,xmin::Int64,xmax::Int64,ymin::Int64,ymax::Int64)
    name = directory*fitsname*".fits"
    #image = read(FitsImage, name)

    f = FITS(name)
    img = read(f[1], xmin:xmax,ymin:ymax)
    return img


end

function main()

    directory = "/home/unahzaal/Documents/StageM2/SINFONI_DHTAUB/reduceddatadirty/DH_TAU_B/06-11-2007/Step4/"
    fitsnamewave = "out_wcal_stack_0000"
    fitsnamefibre = "out_ns_stack_0000"
    
    fibre = SlitletLoad(fitsnamefibre,directory,519,582,5,2000)
    wave = SlitletLoad(fitsnamewave,directory,519,582,5,2000)

    badfibre = Array{Float64, 2}(undef, size(fibre))
    badwave = Array{Float64, 2}(undef, size(wave))

    badfibre .= 1.
    badwave .= 1.

    #bad pixel map
    for i in 1:size(fibre)[1]
        dam = mad(fibre[i,:])
        med = median(fibre[i,:])
        tresh = med + 5*dam
        for j in 1:size(fibre)[2]
            if fibre[i,j] > tresh
                badfibre[i,j] = 0.
            end
        end
    end
    for j in 1:size(wave)[2]
        dam = mad(wave[:,j])
        med = median(wave[:,j])
        tresh = med + 5*dam 
        for i in 1:size(wave)[1]
            if wave[i,j] > tresh
                badwave[i,j] = 0.
            end
        end
    end

    #filtering for finding position
    fibref = mapwindow(median, fibre, (1,11))
    wavef = mapwindow(median, wave, (11,1))
    centerwave = RayFinding(vec(Contrast(wavef,2)),20,10)



    twave = transpose(wave)
    tbadwave = transpose(badwave)

    function pcostfibre(x)
        p  = projection(Val(:proj), 1:64, badfibre, fibre, CatmullRomSpline(),(i,j) ->  i + x[1] + x[2]*1e-3*j + x[3]*1e-6*j*j);
        return -sum(p[[32,33]])
    end
    function pcostwave(x)
        p  = projection(Val(:proj), 1:1994, tbadwave, twave, CatmullRomSpline(),(i,j) ->  i + x[1] + x[2]*1e-3*j + x[3]*1e-6*j*j);
        return -sum(p[centerwave])
    end


 
    x = [0.,0.,0.]
    rhoend = 1e-6

    minfibre = newuoa!(pcostfibre, x, 10.,  rhoend; verbose = 2, maxeval = 5000, check = false)
    minwave = newuoa!(pcostwave, x, 10.,  rhoend; verbose = 2, maxeval = 5000, check = false)

    x = minfibre[2]
    pfibre  = projection(Val(:proj), 1:64, badfibre, fibre, CatmullRomSpline(),(i,j) ->  i + x[1] + x[2]*1e-3*j + x[3]*1e-6*j*j )
    x = minwave[2]
    pwave  = projection(Val(:proj), 1:1994, badwave, wave, CatmullRomSpline(),(i,j) ->  i + x[1] + x[2]*1e-3*j + x[3]*1e-6*j*j)

    x = minfibre[2]
    reffibre = deprojection(pfibre,(1:64,1:1994),CatmullRomSpline(),(i,j) ->  i + x[1] + x[2]*1e-3*j + x[3]*1e-6*j*j)
    f = FITS("ToyModelImages/fibre.fits", "w")
    write(f,fibre)
    close(f)
    f = FITS("ToyModelImages/wave.fits", "w")
    write(f,wave)
    close(f)
    f = FITS("ToyModelImages/fibref.fits", "w")
    write(f,fibref)
    close(f)
    f = FITS("ToyModelImages/wavef.fits", "w")
    write(f,wavef)
    close(f)
    f = FITS("ToyModelImages/reffibre.fits", "w")
    write(f,reffibre.parent)
    close(f)
    f = FITS("ToyModelImages/pfibre.fits", "w")
    write(f,pfibre)
    close(f)
    f = FITS("ToyModelImages/pwave.fits", "w")
    write(f,pwave)
    close(f)
    f = FITS("ToyModelImages/badfibre.fits", "w")
    write(f,badfibre)
    close(f)
    f = FITS("ToyModelImages/badwave.fits", "w")
    write(f,badwave)
    close(f)
    

    println(size(fibre))

    





end

using Revise
using FITSIO, EasyFITS
using OptimPackNextGen.Powell
using StatsBase,Statistics
using InterpolationKernels

function script()

    directory = "/home/unahzaal/Documents/StageM2/SINFONI_DHTAUB/reduceddatadirty/DH_TAU_B/06-11-2007/Step4/"
    fitsnamefibre = "out_ns_stack_0000.fits"
    pathf = directory*fitsnamefibre

    path = "/home/unahzaal/Documents/StageM2/SINFONI_DHTAUB/rawdata/"
    lamp =read(FITS(path*"SINFO.2007-11-07T11:19:43.995.fits")[1])
    fibre =read(FITS(pathf)[1])

    dark = zeros(Float32,size(lamp)...,3)
    dark[:,:,1] = read(FITS(path*"SINFO.2007-11-07T10:18:16.606.fits")[1])
    dark[:,:,2] = read(FITS(path*"SINFO.2007-11-07T10:23:32.890.fits")[1])
    dark[:,:,3] = read(FITS(path*"SINFO.2007-11-07T10:28:54.896.fits")[1])

    good = median(dark) .-  3*mad(dark) .< mean(dark,dims=3)[205:262,:,1] .< median(dark) .+  3*mad(dark)
    medimg = mapwindow(median, median(dark,dims=3), (5,1,1)) .- median(dark,dims=3) 
    good .*= median(medimg) .-  3*mad(medimg) .< median(medimg,dims=3)[205:262,:,1] .< median(medimg) .+  3*mad(medimg)

    lmp = lamp[205:262,:]
    lmp = convert(Matrix{Float64},lmp)
    fbr = fibre[205:262,:]
    fbr = convert(Matrix{Float64},fbr)


    #ker = CatmullRomSpline(Float64)
    #ker = CatmullRomSpline{Float64}() # there is an issue with different version of InterpolationKernels
    ker = BSpline{3,Float64}()
    indices =  -999:3000
    function pcostl(x)
        p  = projection(Val(:proj),indices, good', lmp', ker,(i,j) ->  i + x[1] + x[2]*1e-3*j + x[3]*1e-6*j*j);
        return -sum(p[find_peaks(p; dist=15,nmax=30)])
    end
    function pcostf(x)
        p  = projection(Val(:proj),indices, good, fbr, ker,(i,j) ->  i + x[1] + x[2]*1e-3*j + x[3]*1e-6*j*j);
        return -sum(p[32:33])
    end


    x = [  0., 0., 0.];
    (~,xl) = newuoa(pcostl, x, 1.,  1e-6; verbose = 2, maxeval = 500,check=false)
    (~,xf) = newuoa(pcostf, x, 1.,  1e-6; verbose = 2, maxeval = 500,check=false)

    fcoord(i,j) =  i +  xl[1] + xl[2]*1e-3*j + xl[3]*1e-6*j*j
    fcoordf(i,j) =  i +  xf[1] + xf[2]*1e-3*j + xf[3]*1e-6*j*j

    spectre  = projection(Val(:proj), indices,good', lmp', ker,fcoord );
    spectref  = projection(Val(:proj), indices,good, fbr, ker,fcoordf );

    lampmodel = deprojection(spectre,axes(lmp'), ker,fcoord,indices);
    fibremodel = deprojection(spectref,axes(fbr), ker,fcoordf,indices);


    lampmodelnonfit = deprojection(spectre,axes(lmp'), ker,(i,j)->i,indices);
    fibremodelnonfit = deprojection(spectref,axes(fbr), ker,(i,j)->i,indices);


    tlampmodel = copy(lampmodel')
    tfibremodel = copy(fibremodel')

    f = FITS("ToyModelImages/lampmodel.fits", "w")
    write(f,lampmodel)
    close(f)
    f = FITS("ToyModelImages/lampmodelnonfit.fits", "w")
    write(f,lampmodelnonfit)
    close(f)
    f = FITS("ToyModelImages/lmp.fits", "w")
    write(f,lmp)
    close(f)
    f = FITS("ToyModelImages/reslmp.fits", "w")
    write(f,Residual(lampmodel,lampmodelnonfit))
    close(f)
    f = FITS("ToyModelImages/restrue.fits", "w")
    write(f,Residual(lmp,tlampmodel))
    close(f)
    f = FITS("ToyModelImages/fibremodel.fits", "w")
    write(f,fibremodel)
    close(f)
    f = FITS("ToyModelImages/fibremodelnonfit.fits", "w")
    write(f,fibremodelnonfit)
    close(f)
    f = FITS("ToyModelImages/fbr.fits", "w")
    write(f,fbr)
    close(f)
    f = FITS("ToyModelImages/resfbr.fits", "w")
    write(f,Residual(fibremodel,fibremodelnonfit))
    close(f)
    f = FITS("ToyModelImages/restruef.fits", "w")
    write(f,Residual(fbr,fibremodel))
    close(f)

end