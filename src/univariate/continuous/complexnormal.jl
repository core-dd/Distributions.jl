"""
    ComplexNormal()

The *complex normal distribution* with mean 'μ' and variance 'σ^2' has probability density function

```math
f(x; \\mu, \\sigma) = \\frac{1}{πσ^2} \\exp \\left(-\\frac{1}{σ^2} (x-μ)^\\ast (x-μ) \\right), \\quad x \\in \\mathbb{C}
```

It is related to the Normal distribution via

Andersen et al. - 1995 - The Multivariate Complex Normal Distribution

"""

struct ComplexNormal{T<:Real} <: ContinuousUnivariateDistribution
  μ::Complex{T}
  σ::T
  ComplexNormal{T}(µ::Complex{T}, σ::T) where {T<:Real} = new{T}(µ, σ)
end

function ComplexNormal(μ::Complex{T}, σ::T; check_args=true) where {T <: Real}
    check_args && @check_args(ComplexNormal, σ >= zero(σ))
    return ComplexNormal{T}(μ, σ)
end

ComplexNormal(μ::Real, σ::Real) = ComplexNormal(μ + 1.0im, σ)
ComplexNormal(μ::Complex{Real}, σ::Real) = ComplexNormal(promote(μ, σ)...)
ComplexNormal(μ::Integer, σ::Integer) = ComplexNormal(float(μ) + 1.0im, float(σ))
ComplexNormal(μ::Complex{Integer}, σ::Integer) = ComplexNormal(float(μ), float(σ))
ComplexNormal(μ::T) where {T <: Real} = ComplexNormal(μ+1.0im, one(T))
ComplexNormal(μ::T) where {T <: Complex{Real}} = ComplexNormal(μ, one(T))
ComplexNormal() = ComplexNormal(0.0, 1.0, check_args=false)

# #### Conversions
# TODO:

@distr_support ComplexNormal -Inf-Inf*im Inf+Inf*im  # does that make sence in the grand scheme?

#### Parameters

params(d::ComplexNormal) = (d.μ, d.σ)
@inline partype(d::ComplexNormal{T}) where {T<:Real} = T

location(d::ComplexNormal) = d.μ
scale(d::ComplexNormal) = d.σ

Base.eltype(::Type{ComplexNormal{T}}) where {T} = T

#### Statistics

mean(d::ComplexNormal) = d.μ
#median(d::ComplexNormal) = d.μ # TODO: check
mode(d::ComplexNormal) = d.μ

var(d::ComplexNormal) = abs2(d.σ)
std(d::ComplexNormal) = d.σ
#skewness(d::ComplexNormal{T}) where {T<:Real} = zero(T) # TODO: check
#kurtosis(d::ComplexNormal{T}) where {T<:Real} = zero(T) # TODO: check

#entropy(d::ComplexNormal) = (log2π + 1)/2 + log(d.σ)  # TODO: check

#### Evaluation

# Helpers

# pdf
function pdf(d::ComplexNormal, x::Complex{Real})
  σ2 = d.σ2^2
  1/(π*σ2) * exp(-1/σ2 * abs2(x-d.μ))
end

# logpdf
# logcdf
# logccdf
# cdf
# ccdf
# quantile
# cquantile
# mgf
# cf
cf(d::ComplexNormal, t::Complex{Real}) = exp(1.0im * real(conj(t)*d.μ - d.σ^2 / 4 * abs2(t)))

#### Sampling

rand(rng::AbstractRNG, d::ComplexNormal{T}) where {T} = d.μ + d.σ / sqrt(2) * (randn(rng, T) + one(T)*im * randn(rng, T))

#### Fitting
# TODO:
