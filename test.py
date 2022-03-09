from multiprocessing import Pool
import create_rateCSV


if __name__ == '__main__':
    p = Pool(1)
    create = create_rateCSV.create_rateCSV(year=2022, coin='btc_jpy')
    create.update_csv()
