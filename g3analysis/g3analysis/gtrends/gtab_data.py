
import gtab
import pandas as pd

class GTAB_Handler():

    def __init__(self, geo_id="", timeframe="2004-01-01 2021-01-01", gtab_cat="1209"):

        ## categories
        #Military: 366
        #World News: 1209
        #Law & Government, Public Policy, International Relations: 521

        self.t = gtab.GTAB()
        self.geo_id = geo_id
        self.timeframe = timeframe
        self.gtab_cat = gtab_cat
        self.activate_anchorbank()
        print(f"geo_id: {geo_id}, timeframe: {timeframe}, cat: {gtab_cat}")


    def create_anchorbank(self, num_anchors=100):

        # https://github.com/pat310/google-trends-api/wiki/Google-Trends-Categories
        self.t.set_options(pytrends_config={"geo": self.geo_id, "timeframe": self.timeframe, 'cat': self.gtab_cat},
                           gtab_config={"anchor_candidates_file": "550_cities_and_countries.txt",
                                        "num_anchors": num_anchors})
        self.t.create_anchorbank(verbose=True)  # takes a while to run since it queries Google Trends.
        self.activate_anchorbank()


    def activate_anchorbank(self,):

        #self.t.list_gtabs()
        if self.gtab_cat != None:
            self.t.set_active_gtab(f"google_anchorbank_geo={self.geo_id}_timeframe={self.timeframe}_cat={self.gtab_cat}.tsv")
        else:
            self.t.set_active_gtab(f"google_anchorbank_geo={self.geo_id}_timeframe={self.timeframe}.tsv")


    def query(self, geo_list = []):

        df = pd.DataFrame()
        column_names = list()

        for geo in geo_list:

            df_q = self.t.new_query(geo)
            column_name = "gtab_" + geo
            #column_name = "gtab"
            column_names.append(column_name)
            df_q = df_q.rename(columns={"max_ratio": column_name})[[column_name]]

            if len(df) == 0:
                df = df_q
            else:
                df = pd.merge(df, df_q, left_index=True, right_index=True)
        return df



if __name__ == "__main__":

    g = GTAB_Handler(gtab_cat = 366)#geo_id="US-NJ-609", timeframe="2004-01-01 2020-12-31")
    #g.activate_anchorbank()
    df_q = g.query(["yemen"])