# convert frame to hsv
    hsv = cv.cvtColor(frame_stab, cv.COLOR_RGB2HSV)
    frame_stab = hsv[:, :, 2]  # choose V channel

    #calculate KDE for background and foreground
    x_grid = np.linspace(0, 255, 256)
    kde = gaussian_kde(frame_stab.ravel(), bw_method='silverman')
    kde_pdf = kde.evaluate(x_grid)

    #plot KDE  from all picture
    plt.figure()
    # plot histgram of sample
    #plt.hist(frame_stab.ravel(), bins=256, density = True)
    # plot data generating density
    #plt.plot(x_grid, stats.norm.pdf(x_grid), color="r", label='DGP normal')
    # plot estimated density
    plt.plot(x_grid, kde_pdf, label='kde', color="g")
    plt.title('Kernel Density Estimation')
    plt.legend()
    plt.show()

    print('done')
