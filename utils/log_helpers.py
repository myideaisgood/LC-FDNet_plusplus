

# [Epoch 1400/20000] Y Bitrate    d = 4.0702   b = 3.4705   c = 3.3836
def log_color_info(logging, bitrates, color_names, loc_names, epoch_idx, N_EPOCHS):
    for color in color_names:
        out_string = '[Epoch %d/%d] %s Bitrate '
        out_tuple = (epoch_idx, N_EPOCHS, color)
        for loc in loc_names:
            out_string += '   %s = %.4f = %.4f + %.4f'
            out_tuple += (loc, bitrates[color][loc]['MSB'].avg() + bitrates[color][loc]['LSB'].avg(), bitrates[color][loc]['MSB'].avg(), bitrates[color][loc]['LSB'].avg(),)

        logging.info(out_string % out_tuple)

# [Epoch 1400/20000] Total Bitrate = 8.2486   Y = 3.6414   U = 2.3195   V = 2.2877    LR = 0.0010000
def log_total_info(logging, total_bitrates, color_names, epoch_idx, optimizers, N_EPOCHS):

    out_string = '[Epoch %d/%d] Total Bitrate = %.4f = %.4f + %.4f'
    out_tuple = (epoch_idx, N_EPOCHS)

    total_bit_MSB = 0
    total_bit_LSB = 0
    for color in color_names:
        total_bit_MSB += total_bitrates[color]['MSB'].avg()
        total_bit_LSB += total_bitrates[color]['LSB'].avg()

    total_bit = total_bit_MSB + total_bit_LSB
    out_tuple += (total_bit, total_bit_MSB, total_bit_LSB)

    for color in color_names:
        out_string += '   %s = %.4f = %.4f + %.4f'
        out_tuple += (color, total_bitrates[color]['MSB'].avg() + total_bitrates[color]['LSB'].avg(), total_bitrates[color]['MSB'].avg(), total_bitrates[color]['LSB'].avg())

    # LR
    out_string += '   %s = %.10f'
    out_tuple += ('LR', optimizers['Y']['d']['MSB'].param_groups[0]["lr"])
    
    logging.info(out_string % out_tuple)    

# 0803.png, Y Bitrate    d = 2.3019   b = 1.7269   c = 1.7435
# 0803.png, U Bitrate    d = 2.0224   b = 1.6492   c = 1.6328
# 0803.png, V Bitrate    d = 2.0711   b = 1.6393   c = 1.6481
# 0803.png, Total Bitrate = 5.4784   Y = 1.9241   U = 1.7681   V = 1.7862
def log_img_info(logging, img_name, bitrates, flif_bpp_msb, flif_bpp_lsb, color_names, loc_names):

    # Color Results
    for color in color_names:
        out_string = '%s, %s Bitrate '
        out_tuple = (img_name, color)

        for loc in loc_names:
            out_string += '   %s = %.4f = %.4f + %.4f'
            out_tuple += (loc, bitrates[color][loc]['MSB'].val() + bitrates[color][loc]['LSB'].val(), bitrates[color][loc]['MSB'].val(), bitrates[color][loc]['LSB'].val(),)

        logging.info(out_string % out_tuple)

    # Total Results
    out_string = '%s, Total Bitrate = %.4f = %.4f + %.4f = (%.4f + %.4f) + (%.4f + %.4f)'
    out_tuple = (img_name,)

    color_bits = {}
    for color in color_names:
        color_bit_msb = 0
        color_bit_lsb = 0
        for loc in loc_names:
            color_bit_msb += bitrates[color][loc]['MSB'].val()
            color_bit_lsb += bitrates[color][loc]['LSB'].val()
        color_bits[color] = {}
        color_bits[color]['MSB'] = color_bit_msb
        color_bits[color]['LSB'] = color_bit_lsb

    bpp_msb = flif_bpp_msb + bitrates['total']['MSB'].val()
    bpp_lsb = flif_bpp_lsb + bitrates['total']['LSB'].val()
    total_bpp = bpp_msb + bpp_lsb

    out_tuple += (total_bpp, bpp_msb, bpp_lsb, flif_bpp_msb, bitrates['total']['MSB'].val(), flif_bpp_lsb, bitrates['total']['LSB'].val(), )

    for color in color_names:
        out_string += '   %s = %.4f = %.4f + %.4f'
        out_tuple += (color, color_bits[color]['MSB'] + color_bits[color]['LSB'], color_bits[color]['MSB'], color_bits[color]['LSB'])

    logging.info(out_string % out_tuple)

# Avg BPP Y   d = 8.4285   b = 8.5168   c = 8.3423
# Avg BPP U   d = 9.5238   b = 9.5174   c = 9.5714
# Avg BPP V   d = 9.4804   b = 9.2957   c = 9.2834
# Avg BPP = 27.3199,   Y = 8.4292   U = 9.5375   V = 9.3532
# Avg Enc time = 0.9506   Y = 0.2081   U = 0.3714   V = 0.3712
def log_dataset_info(logging, bitrates, flif_avg_bpp_msb, flif_avg_bpp_lsb, enc_times, flif_avg_time, color_names, loc_names, type='Avg'):

    assert (type=='Avg') or (type=='Best')

    # Avg BPP Y   d = 3.6759   b = 3.0816   c = 3.0361
    for color in color_names:
        out_string = type + ' BPP %s'
        out_tuple = (color, )

        for loc in loc_names:
            out_string += '   %s = %.4f = %.4f + %.4f'
            out_tuple += (loc, bitrates[color][loc]['MSB'].avg() + bitrates[color][loc]['LSB'].avg(), bitrates[color][loc]['MSB'].avg(), bitrates[color][loc]['LSB'].avg())

        logging.info(out_string % out_tuple)

    # Avg BPP = 27.3199,   Y = 8.4292   U = 9.5375   V = 9.3532
    out_string = type + ' BPP = %.4f = %.4f + %.4f = (%.4f + %.4f) + (%.4f + %.4f),'
    
    color_bits = {}
    for color in color_names:
        color_bit_msb = 0
        color_bit_lsb = 0
        for loc in loc_names:
            color_bit_msb += bitrates[color][loc]['MSB'].avg()
            color_bit_lsb += bitrates[color][loc]['LSB'].avg()
        color_bits[color] = {}
        color_bits[color]['MSB'] = color_bit_msb
        color_bits[color]['LSB'] = color_bit_lsb

    bpp_msb = flif_avg_bpp_msb + bitrates['total']['MSB'].avg()
    bpp_lsb = flif_avg_bpp_lsb + bitrates['total']['LSB'].avg()
    bpp_total = bpp_msb + bpp_lsb
    out_tuple = (bpp_total, bpp_msb, bpp_lsb, flif_avg_bpp_msb, bitrates['total']['MSB'].avg(), flif_avg_bpp_lsb, bitrates['total']['LSB'].avg(),)

    for color in color_names:
        out_string += '   %s = %.4f = %.4f + %.4f'
        out_tuple += (color, color_bits[color]['MSB'] + color_bits[color]['LSB'], color_bits[color]['MSB'], color_bits[color]['LSB'])
    
    logging.info(out_string % out_tuple)

    # Avg Enc time = 0.9506   Y = 0.2081   U = 0.3714   V = 0.3712
    out_string = type + ' Enc time = %.4f = %.4f + %.4f'

    total_time = 0
    for loc in loc_names:
        total_time += enc_times[loc].avg()

    out_tuple = (flif_avg_time + total_time, flif_avg_time, total_time)

    for loc in loc_names:
        out_string += '   %s = %.4f'
        out_tuple += (loc, enc_times[loc].avg())
    
    logging.info(out_string % out_tuple)
    