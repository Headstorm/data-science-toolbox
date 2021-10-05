for i in {1..51}
do
    echo processing chunk ${i}
    curl --form addressFile=@./to_geocode/chunk_$i.csv --form benchmark=2020 https://geocoding.geo.census.gov/geocoder/locations/addressbatch --output ./geocoded_addresses/chunk_${i}_result.csv
done