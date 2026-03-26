import ee

# Initialize after authentication
ee.Initialize()

# Example: fetch soil pH at a point
lat, lon = 12.984, 80.235
point = ee.Geometry.Point([lon, lat])

# Load SoilGrids pH dataset (0-5 cm depth)
soil_dataset = ee.Image('OpenLandMap/SOL/SOL_PHH2O_USDA-6A1C_M/v02')

# Sample the point
sample = soil_dataset.sample(region=point, scale=1000).first()

# Get pH value
ph_value = sample.get('phh2o').getInfo() / 10  # divide by 10 for actual pH

print("Estimated soil pH:", ph_value)