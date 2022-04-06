

using DataFitting
using Interpolations, ImageCore, ImageTransformations , TestImages
using CoordinateTransformations, Rotations, StaticArrays
using Images
using Flux
using MultivariatePolynomials

"""
    Poly2D(order::Int,refpix::Tuple{Int,Int},coef::AbstractVector{AbstractVector{Float64}})

Structure containing a 2D polynomial of arbitrary order. 
coef is like : ax^2,bxy,cy^2,dx,ey,f (for Order = 2)
refpix is a reference pixel for mapping
"""
mutable struct Poly2D
    order::Int  # order of the polynomial
    refpix::Tuple{Int,Int}   # reference pixel  
    coef::AbstractVector{AbstractVector{Float64}} #polynomials coeficients  as [ax^2,bxy,cy^2],[dx,ey],[f] (for Order = 2)
end

"""
    (self::Poly2D)(x::Float,y::Float)

compute the value of the polyomial Poly2D at coordinates (x,y)

Example:

D = Poly2D(order,refpix,coef)
z = D(x,y)
"""
function (self::Poly2D)(x::Float64,y::Float64)
    result = 0
    Coef = self.coef
    for i in 0:1:size(Coef)[1]-1
        for j in 0:1:size(Coef[i+1])[1]-1
            result += Coef[i+1][j+1]*x^(size(Coef[i+1])[1]-1-j)*y^(j)
        end
    return result
    end
end

"""
    RandCoef(order::Int)

Generate random coeficient to create a Poly2D object
"""
function RandCoef(order::Int)
    coef = []
    for i in 0:1:order
        push!(coef,rand(Float64,(order-i+1)))
    end
    return coef
end

"""
    Poly2D(order::Int,refpix::Tuple{Int,Int})

Generation of Poly2D object with semi random coeficient
"""
Poly2D(order::Int,refpix::Tuple{Int,Int})=Poly2D(order::Int,refpix::Tuple{Int,Int},RandCoef(order))

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
    Y = warp(image,ϕ,parent(image))

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
    DeformSlitlet(Slitlet, Coeff = rand(Float64,(4,3)))

usage : DeformedSlitlet,Coeff = DeformSlitlet(OriginalSlitlet, [Coeff = rand(Float64,(4,3))])

Return a deformed slitlet using circular permutation. And the coeficients used for deformation 
"""
function DeformSlitlet(Slitlet, Coeff = rand(Float64,(4,3)))


    #Bending functions 
    Bendλ(λ) = (λ^2)*Coeff[1,1]*0 + (λ/100)*Coeff[1,2] + 10Coeff[1,3]
    Bendx(x) = (x^2)*Coeff[3,1]*0 + (x/100)*Coeff[3,2] + Coeff[3,3]

    #Stretching functions
    Stretchλ(λ) = (λ^2)*Coeff[2,1] + λ*Coeff[2,2] + Coeff[2,3]
    Stretchx(x) = (x^2)*Coeff[3,1] + x*Coeff[3,2] + Coeff[3,3]

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


    return Slitlet,Coeff
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


"""
    oldmain()

Obsolete Toy Model
"""
function oldmain()

    FakeFibre, FakeWave = CreateToySlitlet() #Create the slitlets

    #Deform the image with circular permutation and random coeficient
    FakeWaveDist,Coeffs  = DeformSlitlet(FakeWave) 

    #Deform the image with circular permutation and the previous coeficient
    FakeFibreDist,Coeffs  = DeformSlitlet(FakeFibre,Coeffs)

    CenterPos = GaussFitRow(FakeFibre) #Return an array of the position of each gaussian center 

    PolyCoef = PolynomFitRow(PolyArray) #Return the coeficient of the polynomial associated

end

"""
    main()
    
Toy Model
"""
function main()

    FakeFibre, FakeWave = CreateToySlitlet() #Create the slitlets
    
    #Generate 2 2DPolynomials with semi random coefiscients
    Poly1 = Poly2D(3,(0,0))
    Poly2 = Poly2D(3,(0,0))

    #Deform the images with ImageTransformations.warp -->  DOESNT WORK SEND HELP PLS
    DeformedFakeFibre = ImageWarp(FakeFibre,Poly1,Poly2)
    DeformedFakeWave = ImageWarp(FakeWave,Poly1,Poly2)


end