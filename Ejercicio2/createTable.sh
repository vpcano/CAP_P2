    #!/bin/sh

    algorithms=("greyScale_cpu" "greyScale_gpu")
    resolutions=("SD" "HD" "FHD" "4k" "8k")

    # Print the table headers
    printf "%-21s" "" > table.txt
    for res in "${resolutions[@]}"; do
    printf "%-10s" "$res" >> table.txt
    done
    printf "\n" >> table.txt

    for alg in "${algorithms[@]}"; do

        # Print row header
        printf "%-21s" "$alg" >> table.txt

        # Execute the algorithm for each resolution
        for res in "${resolutions[@]}"; do
            img="images/$res.jpg"
            echo "Processing $img file with $alg algorithm..."
            # Extract the processing time from the programs's output
            processing_time=$(./src/"$alg" $img | tail -n 1 | cut -d' ' -f2)
            echo "$img_name processing time: $processing_time"
            # Calculate the number of FPS, this is, the inverse of the processing time
            printf "%-10s" "$processing_time" >> table.txt
        done

        printf "\n" >> table.txt

    done
