from multiprocessing import Pool
import create_rateCSV


if __name__ == '__main__':
    p = Pool(1)
    create = create_rateCSV.create_rateCSV(year=2017, coin='btc_jpy')
    p.map(create.create_CSV, range(1, 13))
