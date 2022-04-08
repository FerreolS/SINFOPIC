

using DataFitting
using Interpolations, ImageCore, ImageTransformations , TestImages
using CoordinateTransformations, Rotations, StaticArrays, OffsetArrays
using Images
using Flux
using ImageShow, TestImages
using LinearAlgebra

"""
    Poly2D(
        order::Int  #order of the polynomial
        refpix::Tuple{Int,Int}   #reference pixel
        coef::Vector{Any} #polynomials coeficients  as [ax^2,bxy,cy^2],[dx,ey],[f] (for Order = 2)
        map::Tuple{AbstractVector{AbstractVector{Int64}},AbstractVector{AbstractVector{Int64}}} #Exponent Map
    )

Structure containing a 2D polynomial of arbitrary order.
coef is like : ax^2,bxy,cy^2,dx,ey,f (for Order = 2)
refpix is a reference pixel for mapping
"""
mutable struct Poly2D
    order::Int  #order of the polynomial
    refpix::Tuple{Int,Int}   #reference pixel
    coef::Vector{Any} #polynomials coeficients  as [ax^2,bxy,cy^2],[dx,ey],[f] (for Order = 2)
    map::Tuple{AbstractVector{Int64},AbstractVector{Int64}} #Exponent Map
end

"""
    Poly2D(order::Int,refpix::Tuple{Int,Int})

Generation of Poly2D object with automatic exponents
"""
Poly2D(order::Int,refpix::Tuple{Int,Int},coef::Vector{Float64})=Poly2D(order::Int,refpix::Tuple{Int,Int}, coef::Vector{Float64},ExpMap(order::Int))



"""
    Poly2D(order::Int,refpix::Tuple{Int,Int})

Generation of Poly2D object with semi random coeficient
"""
Poly2D(order::Int,refpix::Tuple{Int,Int})=Poly2D(order::Int,refpix::Tuple{Int,Int},RandCoef(order))

"""
    RandCoef(order::Int)

Generate random coeficient to create a Poly2D object
"""
function RandCoef(order::Int)
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
function ExpMap(order::Int)
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
function (self::Poly2D)(x::T,y::T) where {T<:Real}
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
    println("inter = $inter mask1 = $mask1 mask2 = $mask2")
    poly.coef .= 0
    poly.coef[inter] .= 1
end

"""
    Residual(image1::Matrix{T},image2::Matrix{T}) where T
Return the residual image between image1 an image2
"""
function Residual(image1::Matrix{T},image2::Matrix{T}) where T
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

    println("inter = $inter mask1 = $mask1 mask2 = $mask2")
    poly.coef .= 0
    poly.coef[inter] .= 1
end

"""
    Contrast(Image::Matrix{Float64},Dim::Int)

Return the constrast of an image squared along the dimension Dim
"""
function Contrast(Image::Matrix{Float64},Dim::Int)
    f(x) = x^2
    return sum(f,Image,dims=Dim)
end



"""
Return a warped tranformed with the 2D Polynomial associated with Poly
"""
function ImageWarp(image::Matrix{Float64},Poly1::Poly2D,Poly2::Poly2D)

    ϕ(x) = (Poly1(x[1],x[2]),Poly2(x[1],x[2]))
    Y = warp(image,ϕ,axes(image),0.)

    return Y
end



"""
    function CreateToySlitlet()

usage : julia> SlitletFibre, SlitletWave = CreateToySlitlet()

Return two fake SINFONI slitlet mimiquing FIBRE and WAVE slitlet
 filled with gaussian simulated signal and added zero margin for deformation space

"""
function CreateToySlitlet()

    #Actual SINFONI slitlet dimensions
    Dimx = 64
    Dimλ = 2048


    #Slitlet's filling function
    σ = 5.
    gauss(x) = exp(-(x)^2/(2σ^2))/(σ*sqrt(2π))


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

    #Adding margin
    Marginx = zeros(Float64,Dimλ,Dimx ÷ 2)
    Marginλ = zeros(Float64,Dimλ ÷ 2,Dimx+Dimx)

    SlitletFibre = [Marginx SlitletFibre Marginx]
    SlitletFibre = [Marginλ;SlitletFibre;Marginλ]

    SlitletWave = [Marginx SlitletWave Marginx]
    SlitletWave = [Marginλ;SlitletWave;Marginλ]


    return SlitletFibre, SlitletWave

end


"""
    DeformSlitlet(Slitlet, coeff = rand(Float64,(4,3)))

usage : DeformedSlitlet,coeff = DeformSlitlet(OriginalSlitlet, [coeff = rand(Float64,(4,3))])

Return a deformed slitlet using circular permutation. And the coeficients used for deformation
"""
function DeformSlitlet(Slitlet, coeff = rand(Float64,(4,3)))


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
    function PolynomFitRow(PolyArray)

Take the center of each row/column and return the polynomial least square fitting of it as a [a b c] matrix.
"""
function PolynomFitRow(PolyArray)

    filter!(x-> 0. <=x<= 128. ,PolyArray) #Filter out spurious values

    polynome(x,a,b,c) = @. (a*x^2+b*x+c)  #Polynomial to be fitted
    params = [0.,0.,0.]                    #Parameters of above polinomial [a,b,c]


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


function swirl(rotation, strength, radius)

    

    x0 = OffsetArrays.center(img)
    r = log(2)*radius/5

    function swirl_map(x::SVector{N}) where N
        xd = x .- x0
        ρ = norm(xd)
        θ = atan(reverse(xd)...)

        # Note that `x == x0 .+ ρ .* reverse(sincos(θ))`
        # swirl adds more rotations to θ based on the distance to center point
        θ̃ = θ + rotation + strength * exp(-ρ/r)

        SVector{N}(x0 .+ ρ .* reverse(sincos(θ̃)))
    end

    warp(img, swirl_map, axes(img))
    
end

function ImageTransTest()
    img = imresize(testimage("cameraman"), (256, 256))
    # Cartesian to Polar
    ρ = norm(y-y0, x-x0)
    θ = atan(y/x)

    # Polar to Cartesian
    y = y0 + ρ*sin(θ)
    x = x0 + ρ*cos(θ)
    preview = ImageShow.gif([swirl(0, 10, radius) for radius in 10:10:150]; fps=5)
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
    main()

Toy Model
"""
function main()

    FakeFibre, FakeWave = CreateToySlitlet() #Create the slitlets

    #Generate 2 2DPolynomials with semi random coefiscients
    Poly1 = Poly2D(1,(0,0),[1.,0.,0.])
    Poly2 = Poly2D(1,(0,0),[0.,1.,0.])
    println(Poly1)
    println(Poly2)
    #Deform the images with ImageTransformations.warp -->  DOESNT WORK SEND HELP PLS
    DeformedFakeFibre = ImageWarp(FakeFibre,Poly1,Poly2)
    #DeformedFakeWave = ImageWarp(FakeWave,Poly1,Poly2)
    println("deformedsum = $(sum(DeformedFakeFibre)), residualsum = $(sum(Residual(FakeFibre,DeformedFakeFibre)))")


end

main()