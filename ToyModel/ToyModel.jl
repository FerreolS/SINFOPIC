

using FITSIO

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

    """
    #Writing results as fits files
    FitsFibre = FITS("FakeFibre.fits","r+")
    write(FitsFibre[1],SlitletFibre)
    close(FitsFibre)

    FitsWave = FITS("FakeWave.fits","r+")
    write(FitsWave[1],SlitletWave)
    close(FitsWave)
    """
    

    return SlitletFibre, SlitletWave

end

function DeformSlitlet(Slitlet, Coeff = rand(Float64,(4,3)))

    #Coeficients of deformation

    #Deformation Functions
    Bendλ(λ) = (λ^2)*Coeff[1,1]*0 + (λ/100)*Coeff[1,2] + 10Coeff[1,3]
    Bendx(x) = (x^2)*Coeff[3,1]*0 + (x/100)*Coeff[3,2] + Coeff[3,3]

    Stretchλ(λ) = (λ^2)*Coeff[2,1] + λ*Coeff[2,2] + Coeff[2,3]
    Stretchx(x) = (x^2)*Coeff[3,1] + x*Coeff[3,2] + Coeff[3,3]

    #Bending horizontally 
    for (x,col) in enumerate(eachcol(Slitlet))
        offset = convert(Int64,round(Bendx(x))) #Column offset in pixel
        circshift!(col, offset)
    end


    #Bending horizontally 
    for (λ,row) in enumerate(eachrow(Slitlet))
        offset = convert(Int64,round(Bendλ(λ))) #Column offset in pixel
        circshift!(row, offset)
    end


    return Slitlet,Coeff
end


FakeFibre, FakeWave = CreateToySlitlet()
using Images
Gray.(FakeWave)
Gray.(FakeFibre)
FakeWaveDist,Coeffs  = DeformSlitlet(FakeWave)
FakeFibreDist,Coeffs  = DeformSlitlet(FakeFibre,Coeffs)
Gray.(FakeWaveDist)
Gray.(FakeFibreDist)

